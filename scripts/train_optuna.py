#!/usr/bin/env python3
"""
Optuna tuning with cross-validation (PR-AUC objective).

- Dataset: CSV with numeric features; target must be binary (0/1)
- CV:
    * If --group-col exists -> GroupKFold
    * Else -> StratifiedKFold
- Pipeline:
    [ColumnTransformer] -> [SMOTE?] -> [Model]
- Models supported: rf (RandomForest), gbm (GradientBoosting)
- Sampler: smote | class_weight | none
    * 'class_weight' applies only to RandomForest
- Objective: maximize PR-AUC (mean across CV folds)

Outputs:
- Prints best trial and CV metrics
- Saves trials CSV and best params JSON under reports/metrics/

Usage (examples):
  python scripts/train_optuna.py \
    --dataset data/processed/dataset_enriched.csv \
    --target sequia \
    --group-col block_id \
    --model rf \
    --sampler smote \
    --n-trials 50

  python scripts/train_optuna.py \
    --dataset data/processed/dataset_enriched.csv \
    --target sequia \
    --model gbm \
    --sampler none \
    --n-trials 80
"""
import argparse
import json
import os
import sys
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

import optuna
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# imbalanced-learn (only if using smote)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ------------- arg parsing -------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning with CV (PR-AUC objective).")
    p.add_argument("--dataset", "-d", default="data/processed/dataset.csv")
    p.add_argument("--target", "-t", default="sequia")
    p.add_argument("--group-col", default=None)
    p.add_argument("--model", choices=["rf", "gbm"], default="rf")
    p.add_argument("--sampler", choices=["none", "smote", "class_weight"], default="smote")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=None, help="Seconds to stop early (optional).")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--outdir", default="reports/metrics")
    return p.parse_args(argv)


# ------------- helpers -------------
def get_feature_columns(df: pd.DataFrame, target: str, extras_to_drop: List[str]) -> List[str]:
    drop_cols = set([target] + extras_to_drop)
    num_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found after exclusions.")
    return num_cols


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


def splitter(y: np.ndarray, groups: Optional[np.ndarray], n_splits: int, random_state: int):
    if groups is not None:
        print(f"→ Using GroupKFold (n_splits={n_splits})")
        return GroupKFold(n_splits=n_splits).split(np.zeros_like(y), y, groups)
    print(f"→ Using StratifiedKFold (n_splits={n_splits}, shuffle=True, rs={random_state})")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(np.zeros_like(y), y)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }


# ------------- main -------------
def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    print(f"→ Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in dataset.")

    y = df[args.target].values
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Target must be binary (0/1).")

    groups = None
    extras_to_drop = []
    if args.group_col:
        if args.group_col not in df.columns:
            raise ValueError(f"Group column '{args.group_col}' not found.")
        groups = df[args.group_col].values
        extras_to_drop.append(args.group_col)

    # Avoid using LAT/LON directly as features
    for maybe_id in ["LAT", "LON"]:
        if maybe_id in df.columns:
            extras_to_drop.append(maybe_id)

    num_cols = get_feature_columns(df, args.target, extras_to_drop)

    # Exclude obvious leakage features (current-month precip and its direct transforms)
    leakage = {
        "PRECTOTCORR", "ppt_pctl",
        "PRECTOTCORR__z", "PRECTOTCORR__roll3m", "PRECTOTCORR__roll6m",
    }
    num_cols = [c for c in num_cols if c not in leakage]
    
    X = df[num_cols].values

    cv_indices = list(splitter(y, groups, args.n_splits, args.random_state))

    def build_pipeline_for_trial(trial: optuna.Trial) -> SkPipeline:
        pre = make_preprocess(num_cols)

        if args.model == "rf":
            # RandomForest search space
            n_estimators = trial.suggest_int("rf_n_estimators", 200, 1000, step=100)
            max_depth = trial.suggest_int("rf_max_depth", 4, 24)
            min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
            max_features = trial.suggest_float("rf_max_features", 0.3, 1.0)
            rf_kwargs = dict(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=-1,
                random_state=args.random_state,
            )
            if args.sampler == "class_weight":
                rf_kwargs["class_weight"] = "balanced"
            clf = RandomForestClassifier(**rf_kwargs)

        elif args.model == "gbm":
            # GradientBoosting search space
            n_estimators = trial.suggest_int("gbm_n_estimators", 100, 800, step=100)
            learning_rate = trial.suggest_float("gbm_learning_rate", 0.01, 0.2, log=True)
            max_depth = trial.suggest_int("gbm_max_depth", 2, 6)
            min_samples_leaf = trial.suggest_int("gbm_min_samples_leaf", 1, 10)
            subsample = trial.suggest_float("gbm_subsample", 0.6, 1.0)
            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=args.random_state,
            )
            # note: class_weight not supported in sklearn GBM

        else:
            raise ValueError("Unsupported model.")

        if args.sampler == "smote":
            pipe = ImbPipeline(steps=[("pre", pre), ("smote", SMOTE(random_state=args.random_state)), ("clf", clf)])
        else:
            pipe = SkPipeline(steps=[("pre", pre), ("clf", clf)])
        return pipe

    def objective(trial: optuna.Trial) -> float:
        pipe = build_pipeline_for_trial(trial)
        pr_scores = []

        for tr_idx, te_idx in cv_indices:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            pipe.fit(X_tr, y_tr)
            prob_te = pipe.predict_proba(X_te)[:, 1]
            pr_scores.append(average_precision_score(y_te, prob_te))

        mean_pr = float(np.mean(pr_scores))
        # you can report more metrics to Optuna's dashboard if desired
        trial.set_user_attr("cv_pr_auc", mean_pr)
        return mean_pr

    study = optuna.create_study(direction="maximize", study_name=f"{args.model}_{args.sampler}_pr-auc")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    # Summarize
    print("\n=== Best trial ===")
    print(f"number: {study.best_trial.number}")
    print(f"value (PR-AUC): {study.best_value:.5f}")
    print("params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Save results
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(args.outdir, f"optuna_{args.model}_{args.sampler}_trials.csv")
    trials_df.to_csv(trials_csv, index=False)

    best_json = os.path.join(args.outdir, f"optuna_{args.model}_{args.sampler}_best.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_value_pr_auc": study.best_value,
                "best_params": study.best_trial.params,
                "model": args.model,
                "sampler": args.sampler,
                "n_splits": args.n_splits,
                "dataset": args.dataset,
                "target": args.target,
                "group_col": args.group_col,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Saved trials CSV: {trials_csv}")
    print(f"✓ Saved best params: {best_json}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
