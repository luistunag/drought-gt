#!/usr/bin/env python3
"""
Train a FINAL calibrated model using best hyperparams from Optuna JSON.

Reads:
  - Optuna best params JSON (from scripts/train_optuna.py)
  - Full dataset CSV

Pipeline:
  [ColumnTransformer] -> [SMOTE?] -> [Model (RF|GBM)] -> CalibratedClassifierCV

Saves artifacts to --outdir:
  - model.joblib        : calibrated model (predict_proba / predict)
  - columns.json        : feature columns used
  - meta.json           : metadata (model, sampler, calibration, params, dataset)

Usage:
  python scripts/train_from_best.py \
    --best-json reports/metrics/optuna_rf_smote_best.json \
    --dataset data/processed/dataset_enriched.csv \
    --target sequia \
    --group-col block_id \
    --sampler smote \
    --calibration isotonic \
    --inner-cv 3 \
    --outdir models/artifacts
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer

# imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ---------------- args ----------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train final calibrated model from Optuna best params.")
    p.add_argument("--best-json", required=True, help="Path to optuna_*_best.json")
    p.add_argument("--dataset", "-d", required=True, help="Path to processed dataset CSV")
    p.add_argument("--target", "-t", default="sequia", help="Binary target column")
    p.add_argument("--group-col", default=None, help="Optional column to drop from features (e.g., block_id)")
    p.add_argument("--sampler", choices=["none", "smote", "class_weight"], default="smote",
                   help="Imbalance handling strategy")
    p.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid",
                   help="Probability calibration method")
    p.add_argument("--inner-cv", type=int, default=3, help="Inner CV folds for calibration")
    p.add_argument("--outdir", default="models/artifacts", help="Output directory for artifacts")
    return p.parse_args(argv)


# ---------------- helpers ----------------
def load_best(best_json_path: str) -> Dict:
    if not os.path.exists(best_json_path):
        raise FileNotFoundError(f"Best JSON not found: {best_json_path}")
    with open(best_json_path, "r", encoding="utf-8") as f:
        best = json.load(f)
    # Expected keys:
    #  - best_params (dict)
    #  - model: "rf" | "gbm"
    #  - sampler: "smote" | "class_weight" | "none"
    if "best_params" not in best or "model" not in best:
        raise ValueError("Invalid best JSON: missing 'best_params' or 'model'.")
    return best


def make_calibrator(pipe, method: str, cv: int) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=pipe, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=pipe, method=method, cv=cv)


def make_preprocess(num_cols: list) -> ColumnTransformer:
    num_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_columns(df: pd.DataFrame, target: str, extras_to_drop: List[str]) -> List[str]:
    drop = set([target] + extras_to_drop)
    num_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found after exclusions.")
    return num_cols


def build_model_from_best(
    model_kind: str,
    best_params: Dict,
    sampler: str,
) -> SkPipeline:
    """
    Construct the base pipeline (pre + [smote?] + model) using best_params.
    - RF params expected keys: rf_n_estimators, rf_max_depth, rf_min_samples_leaf, rf_max_features
    - GBM params expected keys: gbm_n_estimators, gbm_learning_rate, gbm_max_depth, gbm_min_samples_leaf, gbm_subsample
    """
    if model_kind == "rf":
        rf_kwargs = {
            "n_estimators": int(best_params.get("rf_n_estimators", 400)),
            "max_depth": int(best_params.get("rf_max_depth", 12)),
            "min_samples_leaf": int(best_params.get("rf_min_samples_leaf", 2)),
            "max_features": float(best_params.get("rf_max_features", 0.8)),
            "n_jobs": -1,
            "random_state": 42,
        }
        if sampler == "class_weight":
            rf_kwargs["class_weight"] = "balanced"
        clf = RandomForestClassifier(**rf_kwargs)

    elif model_kind == "gbm":
        clf = GradientBoostingClassifier(
            n_estimators=int(best_params.get("gbm_n_estimators", 300)),
            learning_rate=float(best_params.get("gbm_learning_rate", 0.05)),
            max_depth=int(best_params.get("gbm_max_depth", 3)),
            min_samples_leaf=int(best_params.get("gbm_min_samples_leaf", 2)),
            subsample=float(best_params.get("gbm_subsample", 1.0)),
            random_state=42,
        )
        # Note: sklearn GBM does not support class_weight directly

    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    # Preprocess placeholder; we attach real ColumnTransformer later after knowing columns
    pre = ("pre", "DEFER")  # placeholder so we can insert actual preprocessor later

    if sampler == "smote":
        base = ImbPipeline(steps=[pre, ("smote", SMOTE(random_state=42)), ("clf", clf)])
    else:
        base = SkPipeline(steps=[pre, ("clf", clf)])

    return base


# ---------------- main ----------------
def main(argv=None) -> int:
    args = parse_args(argv)

    best = load_best(args.best_json)
    model_kind = best["model"]
    best_params = best["best_params"]

    # You may override sampler via CLI; else fall back to what best JSON used
    sampler = args.sampler or best.get("sampler", "none")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"→ Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found.")
    y = df[args.target].values
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Target must be binary (0/1).")

    # Exclude non-feature columns
    extras_to_drop = []
    if args.group_col and args.group_col in df.columns:
        extras_to_drop.append(args.group_col)
    for geo in ["LAT", "LON"]:
        if geo in df.columns:
            extras_to_drop.append(geo)

    num_cols = get_feature_columns(df, args.target, extras_to_drop)

    # Exclude obvious leakage features (current-month precip and its direct transforms)
    leakage = {
        "PRECTOTCORR", "ppt_pctl",
        "PRECTOTCORR__z", "PRECTOTCORR__roll3m", "PRECTOTCORR__roll6m",
    }
    num_cols = [c for c in num_cols if c not in leakage]
    
    X = df[num_cols]
    print(f"→ Using {len(num_cols)} numeric features.")

    # Build base pipeline with best params
    base = build_model_from_best(model_kind=model_kind, best_params=best_params, sampler=sampler)

    # Inject real preprocessor (now that we know num_cols)
    pre = make_preprocess(num_cols)
    # Replace placeholder "pre" step
    steps = base.steps
    steps = [("pre", pre)] + [s for s in steps if s[0] != "pre"]
    if isinstance(base, ImbPipeline):
        base = ImbPipeline(steps=steps)
    else:
        base = SkPipeline(steps=steps)

    # Wrap with calibration
    calibrated = make_calibrator(
        base,
        method=args.calibration,
        cv=args.inner_cv,
    )

    print(f"→ Fitting FULL calibrated model (model={model_kind}, sampler={sampler}, calibration={args.calibration}, inner-cv={args.inner_cv})")
    calibrated.fit(X, y)

    # Save artifacts
    model_path = os.path.join(args.outdir, "model.joblib")
    cols_path = os.path.join(args.outdir, "columns.json")
    meta_path = os.path.join(args.outdir, "meta.json")

    joblib.dump(calibrated, model_path)
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": num_cols}, f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_kind,
                "sampler": sampler,
                "calibration": args.calibration,
                "inner_cv": args.inner_cv,
                "best_value_pr_auc": best.get("best_value_pr_auc"),
                "best_params": best_params,
                "dataset": args.dataset,
                "target": args.target,
                "group_col": args.group_col,
            },
            f,
            indent=2,
        )

    print(f"✓ Saved model:   {model_path}")
    print(f"✓ Saved columns: {cols_path}")
    print(f"✓ Saved meta:    {meta_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
