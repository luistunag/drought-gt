#!/usr/bin/env python3
"""
CV training with class imbalance handling + probability calibration.

- Reads a processed dataset (default: data/processed/dataset.csv)
- Target must be binary (0/1), default: 'sequia'
- Features: numeric-only (auto-detected after excluding ids/group/target)
- Outer CV:
    * If --group-col exists -> GroupKFold
    * Else -> StratifiedKFold
- Pipeline:
    [ColumnTransformer] -> [SMOTE?] -> [RandomForest]
- Calibration (inner CV on each outer training fold):
    CalibratedClassifierCV(base_estimator=<pipeline>, method=('sigmoid'|'isotonic'), cv=K)

- Outputs: fold-wise calibrated metrics + CV mean±std (PR-AUC, ROC-AUC, F1, recall, precision)

Usage (examples):
  python scripts/train_calibrated.py \
    --dataset data/processed/dataset_enriched.csv \
    --target sequia \
    --group-col block_id \
    --sampler smote \
    --calibration isotonic \
    --inner-cv 3

  python scripts/train_calibrated.py \
    --dataset data/processed/dataset_enriched.csv \
    --target sequia \
    --sampler class_weight \
    --calibration sigmoid
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer

# imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ---------- args ----------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV training with imbalance + probability calibration.")
    p.add_argument("--dataset", "-d", default="data/processed/dataset.csv", help="Path to CSV dataset.")
    p.add_argument("--target", "-t", default="sequia", help="Binary target column name.")
    p.add_argument("--group-col", default=None, help="Optional grouping column for GroupKFold.")
    p.add_argument("--n-splits", type=int, default=5, help="Outer CV folds.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")

    # Imbalance handling
    p.add_argument(
        "--sampler",
        choices=["none", "smote", "class_weight"],
        default="smote",
        help="Imbalance strategy.",
    )

    # Calibration
    p.add_argument(
        "--calibration",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method.",
    )
    p.add_argument("--inner-cv", type=int, default=3, help="Inner CV folds for calibration.")

    # RF knobs
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    return p.parse_args(argv)


# ---------- helpers ----------
def make_calibrator(pipe, method: str, cv: int) -> CalibratedClassifierCV:
    """Compat: sklearn >=1.5 usa `estimator`; <=1.4 usa `base_estimator`."""
    try:
        return CalibratedClassifierCV(estimator=pipe, method=method, cv=cv)  # new API
    except TypeError:
        return CalibratedClassifierCV(base_estimator=pipe, method=method, cv=cv)  # old API

    
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


def build_base_pipeline(
    num_cols: List[str],
    random_state: int,
    sampler: str,
    n_estimators: int,
    min_samples_leaf: int,
) -> SkPipeline:
    pre = make_preprocess(num_cols)

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


def splitter(y: np.ndarray, groups: Optional[np.ndarray], n_splits: int, random_state: int):
    if groups is not None:
        print(f"→ Using GroupKFold (outer n_splits={n_splits})")
        return GroupKFold(n_splits=n_splits).split(np.zeros_like(y), y, groups)
    print(f"→ Using StratifiedKFold (outer n_splits={n_splits}, shuffle=True, rs={random_state})")
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


# ---------- main ----------
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

    # Avoid leaking geo coordinates into features
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
    
    X = df[num_cols]

    fold_metrics = []

    # Outer CV (OOF evaluation with calibrated probabilities)
    for fold, (tr_idx, te_idx) in enumerate(
        splitter(y, groups, args.n_splits, args.random_state), start=1
    ):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Base pipeline (pre + [smote?] + RF)
        base_pipe = build_base_pipeline(
            num_cols=num_cols,
            random_state=args.random_state,
            sampler=args.sampler,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
        )

        # Wrap with calibration
        # CalibratedClassifierCV will:
        #  - split the *training fold* into inner training/calibration folds
        #  - fit base_pipe on inner-train (SMOTE only on inner-train)
        #  - calibrate on inner-calibration
        #  - final model is an ensemble of calibrated clones (predict_proba averaged)
        calibrated = make_calibrator(
            base_pipe,
            method=args.calibration,
            cv=args.inner_cv,  # inner folds for calibration
        )

        # Fit on outer training fold, evaluate on outer test fold
        calibrated.fit(X_tr, y_tr)
        prob_te = calibrated.predict_proba(X_te)[:, 1]
        pred_te = (prob_te >= 0.5).astype(int)

        m = compute_metrics(y_te, prob_te, pred_te)
        fold_metrics.append(m)
        print(
            f"[fold {fold}] "
            f"PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Precision={m['precision']:.4f}"
        )

    # CV summary
    keys = ["pr_auc", "roc_auc", "f1", "recall", "precision"]
    mu = {k: float(np.mean([d[k] for d in fold_metrics])) for k in keys}
    sd = {k: float(np.std([d[k] for d in fold_metrics])) for k in keys}

    print("\n=== Calibrated CV Summary (mean ± std) ===")
    for k in keys:
        print(f"{k:>9}: {mu[k]:.4f} ± {sd[k]:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
