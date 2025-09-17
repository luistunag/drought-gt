#!/usr/bin/env python3
"""
Training with class imbalance handling (SMOTE inside a pipeline).

- Reads a processed dataset (default: data/processed/dataset.csv)
- Binary target (default: 'sequia')
- Numeric features only (auto-detected after dropping ids/group cols)
- CV:
    * If --group-col exists -> GroupKFold
    * Else -> StratifiedKFold
- Pipeline:
    [ColumnTransformer] -> [SMOTE] -> [Classifier]
  SMOTE is applied only on training folds via imblearn.Pipeline (no leakage).

- Options:
    --sampler none|smote|class_weight
      * none          : no sampler; clf without class_weight
      * smote         : uses SMOTE()
      * class_weight  : no SMOTE; clf with class_weight='balanced'

- Model: RandomForestClassifier (default params, can adjust via flags)
- Metrics: PR-AUC, ROC-AUC, F1, recall, precision (fold-wise + CV summary)

Usage:
  python scripts/train_smote.py \
    --dataset data/processed/dataset_enriched.csv \
    --target sequia \
    --group-col block_id \
    --sampler smote
"""
import argparse
import os
import sys
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
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
from sklearn.ensemble import RandomForestClassifier

# imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV training with imbalance handling (SMOTE/class_weight).")
    p.add_argument("--dataset", "-d", default="data/processed/dataset.csv")
    p.add_argument("--target", "-t", default="sequia")
    p.add_argument("--group-col", default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--sampler",
        choices=["none", "smote", "class_weight"],
        default="smote",
        help="Imbalance handling strategy.",
    )
    # RF knobs (optional)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    return p.parse_args(argv)


def get_feature_columns(df: pd.DataFrame, target: str, extras_to_drop: List[str]) -> List[str]:
    drop_cols = set([target] + extras_to_drop)
    num_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found after exclusions.")
    return num_cols


def build_pipeline(
    num_cols: List[str],
    random_state: int,
    sampler: str,
    n_estimators: int,
    min_samples_leaf: int,
) -> SkPipeline:
    pre = make_preprocess(num_cols)

    # Base classifier (class_weight for 'class_weight' mode only)
    clf_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
    )
    if sampler == "class_weight":
        clf_kwargs["class_weight"] = "balanced"

    clf = RandomForestClassifier(**clf_kwargs)

    if sampler == "smote":
        # SMOTE applied only on training folds via imblearn.Pipeline
        pipe = ImbPipeline(
            steps=[
                ("pre", pre),
                ("smote", SMOTE(random_state=random_state)),
                ("clf", clf),
            ]
        )
    else:
        # 'none' or 'class_weight' use a regular sklearn Pipeline (no sampler)
        pipe = SkPipeline(steps=[("pre", pre), ("clf", clf)])

    return pipe


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

    # Avoid leaking lat/lon as features
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

    pipe = build_pipeline(
        num_cols=num_cols,
        random_state=args.random_state,
        sampler=args.sampler,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
    )

    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(
        splitter(y, groups, args.n_splits, args.random_state), start=1
    ):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipe.fit(X_tr, y_tr)
        prob_te = pipe.predict_proba(X_te)[:, 1]
        pred_te = (prob_te >= 0.5).astype(int)

        m = compute_metrics(y_te, prob_te, pred_te)
        fold_metrics.append(m)
        print(
            f"[fold {fold}] "
            f"PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
            f"F1={m['f1']:.4f} R={m['recall']:.4f} P={m['precision']:.4f}"
        )

    # CV summary
    mu = {k: float(np.mean([d[k] for d in fold_metrics])) for k in fold_metrics[0].keys()}
    sd = {k: float(np.std([d[k] for d in fold_metrics])) for k in fold_metrics[0].keys()}

    print("\n=== CV Summary (mean ± std) ===")
    for k in ["pr_auc", "roc_auc", "f1", "recall", "precision"]:
        print(f"{k:>9}: {mu[k]:.4f} ± {sd[k]:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
