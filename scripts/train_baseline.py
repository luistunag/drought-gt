#!/usr/bin/env python3
"""
Baseline training script (no tuning) with cross-validation.

- Reads a processed dataset (default: data/processed/dataset.csv)
- Target column defaults to 'sequia' (override with --target)
- Uses numeric features only (auto-detected)
- CV:
    * If --group-col exists in the dataset, uses GroupKFold
    * Else uses StratifiedKFold
- Model: RandomForestClassifier(class_weight='balanced')
- Metrics: PR-AUC (primary), ROC-AUC, F1, recall, precision

Usage:
  python scripts/train_baseline.py \
      --dataset data/processed/dataset.csv \
      --target sequia \
      --group-col region_climatica   # optional

Requires: pandas, scikit-learn, numpy
"""
import argparse
import os
import sys
from typing import List, Optional, Tuple, Dict

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline CV training (no tuning).")
    p.add_argument(
        "--dataset",
        "-d",
        default="data/processed/dataset.csv",
        help="Path to processed dataset CSV.",
    )
    p.add_argument(
        "--target",
        "-t",
        default="sequia",
        help="Target column name (binary: 0/1).",
    )
    p.add_argument(
        "--group-col",
        default=None,
        help="Optional grouping column for GroupKFold (e.g., 'region_climatica').",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return p.parse_args(argv)


def get_feature_columns(df: pd.DataFrame, target: str, extras_to_drop: List[str]) -> List[str]:
    drop_cols = set([target] + extras_to_drop)
    # Numeric-only features to keep it simple and robust
    num_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found after exclusions.")
    return num_cols


def build_pipeline(num_cols: List[str], random_state: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", clf),
        ]
    )
    return pipe


def cv_splitter(
    y: np.ndarray, groups: Optional[np.ndarray], n_splits: int, random_state: int
):
    if groups is not None:
        print(f"→ Using GroupKFold (n_splits={n_splits})")
        return GroupKFold(n_splits=n_splits).split(X=np.zeros_like(y), y=y, groups=groups)
    else:
        print(f"→ Using StratifiedKFold (n_splits={n_splits}, shuffle=True, rs={random_state})")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return skf.split(X=np.zeros_like(y), y=y)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # y_prob: positive-class probabilities
    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    print(f"→ Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available: {list(df.columns)[:20]}...")

    y = df[args.target].values
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Target must be binary (0/1).")

    groups = None
    extras_to_drop = []
    if args.group_col:
        if args.group_col not in df.columns:
            raise ValueError(f"Group column '{args.group_col}' not found in dataset.")
        groups = df[args.group_col].values
        extras_to_drop.append(args.group_col)

    # Also drop typical geospatial IDs from features if present (they can leak spatial info)
    for maybe_id in ["LAT", "LON"]:
        if maybe_id in df.columns:
            extras_to_drop.append(maybe_id)

    num_cols = get_feature_columns(df, args.target, extras_to_drop)
    print(f"→ Using {len(num_cols)} numeric features.")

    # Exclude obvious leakage features (current-month precip and its direct transforms)
    leakage = {
        "PRECTOTCORR", "ppt_pctl",
        "PRECTOTCORR__z", "PRECTOTCORR__roll3m", "PRECTOTCORR__roll6m",
    }
    num_cols = [c for c in num_cols if c not in leakage]

    X = df[num_cols].values

    pipe = build_pipeline(num_cols=num_cols, random_state=args.random_state)

    metrics_list = []
    for fold, (tr_idx, te_idx) in enumerate(
        cv_splitter(y=y, groups=groups, n_splits=args.n_splits, random_state=args.random_state),
        start=1,
    ):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipe.fit(X_tr, y_tr)
        prob_te = pipe.predict_proba(X_te)[:, 1]
        pred_te = (prob_te >= 0.5).astype(int)

        m = compute_metrics(y_te, prob_te, pred_te)
        metrics_list.append(m)
        print(
            f"[fold {fold}] "
            f"PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Precision={m['precision']:.4f}"
        )

    # Aggregate
    agg = {k: (np.mean([d[k] for d in metrics_list]), np.std([d[k] for d in metrics_list])) for k in metrics_list[0].keys()}
    print("\n=== CV Summary (mean ± std) ===")
    for k, (mu, sd) in agg.items():
        print(f"{k:>9}: {mu:.4f} ± {sd:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
