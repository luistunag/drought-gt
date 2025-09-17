#!/usr/bin/env python3
"""
Cross-validated threshold selection (no tuning).

- Trains a baseline RF (same as train_baseline.py) with CV
- Searches the probability threshold in [0.05..0.95] that maximizes a chosen metric
- Reports per-fold and mean±std metrics at the selected threshold
- Metrics supported: f1 (default), recall, precision, youden (tpr - fpr), cost (weighted)

Optional cost weighting: cost = w_recall*recall + w_precision*precision (set via --w-recall/--w-precision)

Usage:
  python scripts/threshold_opt.py \
    --dataset data/processed/dataset_enriched.csv \
    --target sequia \
    --group-col block_id \
    --metric f1 \
    --n-splits 5
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
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV threshold optimization.")
    p.add_argument("--dataset", "-d", default="data/processed/dataset.csv")
    p.add_argument("--target", "-t", default="sequia")
    p.add_argument("--group-col", default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--metric",
        choices=["f1", "recall", "precision", "youden", "cost"],
        default="f1",
        help="Objective to maximize when selecting threshold.",
    )
    p.add_argument("--w-recall", type=float, default=0.7, help="Cost weight for recall (when --metric cost).")
    p.add_argument("--w-precision", type=float, default=0.3, help="Cost weight for precision (when --metric cost).")
    p.add_argument("--min-th", type=float, default=0.05)
    p.add_argument("--max-th", type=float, default=0.95)
    p.add_argument("--step", type=float, default=0.01)
    return p.parse_args(argv)


def get_feature_columns(df: pd.DataFrame, target: str, extras_to_drop: List[str]) -> List[str]:
    drop_cols = set([target] + extras_to_drop)
    num_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found after exclusions.")
    return num_cols


def build_pipeline(num_cols: List[str], random_state: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)],
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
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def splitter(y: np.ndarray, groups: Optional[np.ndarray], n_splits: int, random_state: int):
    if groups is not None:
        print(f"→ Using GroupKFold (n_splits={n_splits})")
        return GroupKFold(n_splits=n_splits).split(np.zeros_like(y), y, groups)
    print(f"→ Using StratifiedKFold (n_splits={n_splits}, shuffle=True, rs={random_state})")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(np.zeros_like(y), y)


def metric_value(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                 w_recall: float, w_precision: float) -> float:
    if name == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if name == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    if name == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if name == "youden":
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return float(np.max(tpr - fpr))
    if name == "cost":
        # Simple weighted combination (higher is better)
        r = recall_score(y_true, y_pred, zero_division=0)
        p = precision_score(y_true, y_pred, zero_division=0)
        return w_recall * r + w_precision * p
    raise ValueError(f"Unknown metric: {name}")


def compute_standard_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

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

    for maybe_id in ["LAT", "LON"]:
        if maybe_id in df.columns:
            extras_to_drop.append(maybe_id)

    num_cols = get_feature_columns(df, args.target, extras_to_drop)
    X = df[num_cols].values
    pipe = build_pipeline(num_cols=num_cols, random_state=args.random_state)

    thresholds = np.arange(args.min_th, args.max_th + 1e-9, args.step)
    fold_rows = []
    best_thresholds = []

    for fold, (tr_idx, te_idx) in enumerate(
        splitter(y, groups, args.n_splits, args.random_state), start=1
    ):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipe.fit(X_tr, y_tr)
        prob_te = pipe.predict_proba(X_te)[:, 1]

        # Search best threshold on this fold
        best_th, best_score = 0.5, -np.inf
        for th in thresholds:
            pred_te = (prob_te >= th).astype(int)
            score = metric_value(args.metric, y_te, pred_te, prob_te, args.w_recall, args.w_precision)
            if score > best_score:
                best_score = score
                best_th = th

        best_thresholds.append(best_th)
        pred_best = (prob_te >= best_th).astype(int)
        m = compute_standard_metrics(y_te, prob_te, pred_best)

        tn, fp, fn, tp = confusion_matrix(y_te, pred_best).ravel()
        fold_rows.append({
            "fold": fold,
            "best_threshold": best_th,
            **m,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
        print(
            f"[fold {fold}] th*={best_th:.2f} | "
            f"PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
            f"F1={m['f1']:.4f} R={m['recall']:.4f} P={m['precision']:.4f} "
            f"TP={tp} FP={fp} TN={tn} FN={fn}"
        )

    # Aggregate
    best_th_mean = float(np.mean(best_thresholds))
    best_th_std = float(np.std(best_thresholds))
    print(f"\n=== Selected thresholds across folds (mean ± std): {best_th_mean:.3f} ± {best_th_std:.3f}")

    df_cv = pd.DataFrame(fold_rows)
    agg = df_cv[["pr_auc", "roc_auc", "f1", "recall", "precision"]].mean().to_dict()
    std = df_cv[["pr_auc", "roc_auc", "f1", "recall", "precision"]].std().to_dict()
    print("\n=== CV Summary (mean ± std) @ best thresholds ===")
    for k in ["pr_auc", "roc_auc", "f1", "recall", "precision"]:
        print(f"{k:>9}: {agg[k]:.4f} ± {std[k]:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
