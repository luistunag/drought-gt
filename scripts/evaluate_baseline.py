#!/usr/bin/env python3
"""
Baseline evaluation with cross-validation (out-of-fold predictions).

- Reads processed dataset (default: data/processed/dataset.csv)
- Target defaults to 'sequia' (override with --target)
- CV:
    * If --group-col exists -> GroupKFold
    * Else -> StratifiedKFold
- Model: RandomForestClassifier(class_weight='balanced')
- Outputs:
    * reports/figures/pr_curve.png
    * reports/figures/roc_curve.png
    * reports/metrics/cv_metrics.csv
    * prints CV mean ± std for PR-AUC, ROC-AUC, F1, Recall, Precision

Usage:
  python scripts/evaluate_baseline.py \
      --dataset data/processed/dataset.csv \
      --target sequia \
      --group-col region_climatica   # optional

Requires: pandas, scikit-learn, numpy, matplotlib
"""
import argparse
import os
import sys
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline CV evaluation (no tuning).")
    p.add_argument("--dataset", "-d", default="data/processed/dataset.csv")
    p.add_argument("--target", "-t", default="sequia")
    p.add_argument("--group-col", default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--outdir", default="reports")
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
        return GroupKFold(n_splits=n_splits).split(X=np.zeros_like(y), y=y, groups=groups)
    print(f"→ Using StratifiedKFold (n_splits={n_splits}, shuffle=True, rs={random_state})")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(X=np.zeros_like(y), y=y)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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

    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "metrics"), exist_ok=True)

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

    for maybe_id in ["LAT", "LON"]:
        if maybe_id in df.columns:
            extras_to_drop.append(maybe_id)

    num_cols = get_feature_columns(df, args.target, extras_to_drop)
    X = df[num_cols].values
    pipe = build_pipeline(num_cols=num_cols, random_state=args.random_state)

    # Out-of-fold predictions
    oof_prob = np.zeros_like(y, dtype=float)
    oof_pred = np.zeros_like(y, dtype=int)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(
        splitter(y, groups, args.n_splits, args.random_state), start=1
    ):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipe.fit(X_tr, y_tr)
        prob_te = pipe.predict_proba(X_te)[:, 1]
        pred_te = (prob_te >= 0.5).astype(int)

        oof_prob[te_idx] = prob_te
        oof_pred[te_idx] = pred_te

        m = compute_metrics(y_te, prob_te, pred_te)
        fold_metrics.append({"fold": fold, **m})
        print(
            f"[fold {fold}] PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
            f"F1={m['f1']:.4f} Recall={m['recall']:.4f} Precision={m['precision']:.4f}"
        )

    # Aggregate CV metrics
    agg = {
        k: (np.mean([d[k] for d in fold_metrics]), np.std([d[k] for d in fold_metrics]))
        for k in ["pr_auc", "roc_auc", "f1", "recall", "precision"]
    }
    print("\n=== CV Summary (mean ± std) ===")
    for k, (mu, sd) in agg.items():
        print(f"{k:>9}: {mu:.4f} ± {sd:.4f}")

    # Save metrics CSV
    pd.DataFrame(fold_metrics).to_csv(
        os.path.join(args.outdir, "metrics", "cv_metrics.csv"), index=False
    )

    # PR curve (OOF)
    prec, rec, _ = precision_recall_curve(y, oof_prob)
    ap = average_precision_score(y, oof_prob)
    plt.figure()
    plt.step(rec, prec, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (OOF) — AP={ap:.3f}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "figures", "pr_curve.png"), dpi=160)
    plt.close()

    # ROC curve (OOF)
    fpr, tpr, _ = roc_curve(y, oof_prob)
    roc = roc_auc_score(y, oof_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC curve (OOF) — AUC={roc:.3f}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "figures", "roc_curve.png"), dpi=160)
    plt.close()

    print(f"\nSaved figures to: {os.path.join(args.outdir, 'figures')}")
    print(f"Saved metrics to: {os.path.join(args.outdir, 'metrics', 'cv_metrics.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
