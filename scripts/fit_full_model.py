#!/usr/bin/env python3
"""
Fit a calibrated model on the FULL dataset and save artifacts.

Pipeline:
  [ColumnTransformer] -> [SMOTE?] -> [RandomForest] -> CalibratedClassifierCV

Artifacts saved to --outdir (default: models/artifacts):
  - model.joblib        : calibrated pipeline (ready for predict_proba / predict)
  - columns.json        : feature column names used
  - meta.json           : metadata (target, sampler, calibration, params)

Usage:
  python scripts/fit_full_model.py \
    --dataset data/processed/dataset_enriched.csv \
    --target sequia \
    --group-col block_id \  # optional (ignored for fitting, only for dropping from features)
    --sampler smote \
    --calibration isotonic \
    --inner-cv 3
"""
import argparse
import json
import os
import sys
from typing import List, Optional, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer

# imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit calibrated model on full dataset and save artifacts.")
    p.add_argument("--dataset", "-d", default="data/processed/dataset.csv")
    p.add_argument("--target", "-t", default="sequia")
    p.add_argument("--group-col", default=None, help="Optional column to drop from features (e.g., block_id).")
    p.add_argument("--outdir", default="models/artifacts")

    # Imbalance
    p.add_argument("--sampler", choices=["none", "smote", "class_weight"], default="smote")

    # Calibration
    p.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--inner-cv", type=int, default=3)

    # RF knobs
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args(argv)


def make_calibrator(pipe, method: str, cv: int) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=pipe, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=pipe, method=method, cv=cv)
    

def get_feature_columns(df: pd.DataFrame, target: str, extras_to_drop: List[str]) -> List[str]:
    drop_cols = set([target] + extras_to_drop)
    num_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found after exclusions.")
    return num_cols


def build_base_pipeline(
    num_cols: List[str],
    sampler: str,
    n_estimators: int,
    min_samples_leaf: int,
    random_state: int,
) -> SkPipeline:
    pre = make_preprocess(num_cols)

    rf_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
    )
    if sampler == "class_weight":
        rf_kwargs["class_weight"] = "balanced"

    clf = RandomForestClassifier(**rf_kwargs)

    if sampler == "smote":
        pipe = ImbPipeline(
            steps=[
                ("pre", pre),
                ("smote", SMOTE(random_state=random_state)),
                ("clf", clf),
            ]
        )
    else:
        pipe = SkPipeline(steps=[("pre", pre), ("clf", clf)])

    return pipe

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


def main(argv=None) -> int:
    args = parse_args(argv)

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

    extras_to_drop = []
    # Drop grouping and geo IDs from features if present
    if args.group_col and args.group_col in df.columns:
        extras_to_drop.append(args.group_col)
    for maybe_id in ["LAT", "LON"]:
        if maybe_id in df.columns:
            extras_to_drop.append(maybe_id)

    num_cols = get_feature_columns(df, args.target, extras_to_drop)
    X = df[num_cols]

    print(f"→ Feature columns: {len(num_cols)}")

    # Base pipeline with (optional) SMOTE / class_weight
    base_pipe = build_base_pipeline(
        num_cols=num_cols,
        sampler=args.sampler,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    # Wrap with calibration (inner CV)
    model = make_calibrator(
        base_pipe,
        method=args.calibration,
        cv=args.inner_cv,
    )

    print(f"→ Fitting FULL model with sampler={args.sampler}, calibration={args.calibration} (inner-cv={args.inner_cv})")
    model.fit(X, y)

    # Save artifacts
    model_path = os.path.join(args.outdir, "model.joblib")
    cols_path = os.path.join(args.outdir, "columns.json")
    meta_path = os.path.join(args.outdir, "meta.json")

    joblib.dump(model, model_path)
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": num_cols}, f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": args.target,
                "sampler": args.sampler,
                "calibration": args.calibration,
                "inner_cv": args.inner_cv,
                "n_estimators": args.n_estimators,
                "min_samples_leaf": args.min_samples_leaf,
                "random_state": args.random_state,
                "dataset": args.dataset,
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
