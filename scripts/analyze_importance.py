#!/usr/bin/env python3
"""
Analyze feature importance via permutation importance on the trained pipeline.

Inputs:
  - Trained model pipeline: models/artifacts/model.joblib
  - Feature list:           models/artifacts/columns.json
  - Dataset (CSV):          data/processed/dataset_selected.csv (or target)

Outputs:
  reports/metrics/importance_permutation.csv
  reports/figures/importance_permutation_ap.png
  reports/figures/importance_permutation_roc.png

Notes:
  - Works with any sklearn/imbalanced-learn pipeline (calibrated or not).
  - Evaluates importance on a group-aware subset (no refit). This is fast and
    generally informative, though not as "pure" as CV re-fitting.
"""

import argparse
import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Permutation importance of the trained model.")
    p.add_argument("--dataset", default="data/processed/dataset_selected.csv")
    p.add_argument("--target", default="sequia")
    p.add_argument("--group-col", default="block_id")
    p.add_argument("--model", default="models/artifacts/model.joblib")
    p.add_argument("--columns", default="models/artifacts/columns.json")
    p.add_argument("--score", choices=["ap", "roc", "both"], default="both",
                   help="Metric for permutation importance: average precision (ap), ROC-AUC (roc), or both.")
    p.add_argument("--subsample-frac", type=float, default=0.35,
                   help="Fraction of rows to evaluate importance on (for speed).")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=10, help="Permutation repeats per feature.")
    p.add_argument("--topk", type=int, default=40, help="Top-k features to plot.")
    return p.parse_args()


def ensure_dirs():
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)


def load_columns(columns_path: str) -> List[str]:
    with open(columns_path, "r") as f:
        data = json.load(f)
    # Supports either {"feature_columns":[...]} or a raw list
    if isinstance(data, dict) and "feature_columns" in data:
        return list(data["feature_columns"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unrecognized columns format in {columns_path}")


def pick_subset(df: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series],
                frac: float, seed: int) -> pd.DataFrame:
    if frac >= 0.999:
        return df, y, groups
    if groups is not None and groups.nunique() > 1:
        # sample by groups to avoid splitting the same group across train/test
        uniq = groups.dropna().unique()
        rng = np.random.default_rng(seed)
        keep_groups = rng.choice(uniq, size=max(1, int(len(uniq) * frac)), replace=False)
        mask = groups.isin(keep_groups)
        return df.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), groups.loc[mask].reset_index(drop=True)
    # fallback: stratified by target
    X_sub, _, y_sub, _ = train_test_split(df, y, test_size=(1.0 - frac),
                                          random_state=seed, stratify=y)
    if groups is not None:
        groups = groups.loc[X_sub.index]
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True), (groups.reset_index(drop=True) if groups is not None else None)


def barplot(df_imp: pd.DataFrame, score_tag: str, topk: int, out_png: str):
    top = df_imp.head(topk)
    plt.figure(figsize=(9, max(4, 0.3 * len(top))))
    plt.barh(range(len(top)), top["importance_mean"].values, xerr=top["importance_std"].values)
    plt.yticks(range(len(top)), top["feature"].values)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Permutation importance ({score_tag}) - mean ± std")
    plt.title(f"Top {len(top)} features by permutation importance ({score_tag})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def compute_perm_importance(estimator, X: pd.DataFrame, y: pd.Series,
                            scoring: str, n_repeats: int, seed: int) -> pd.DataFrame:
    """
    scoring can be 'average_precision' or 'roc_auc' or a scorer callable.
    """
    result = permutation_importance(
        estimator, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    out = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return out


def main():
    args = parse_args()
    ensure_dirs()

    # Load artifacts
    pipe = joblib.load(args.model)
    feat_cols = load_columns(args.columns)

    # Load data
    df = pd.read_csv(args.dataset, low_memory=False)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in {args.dataset}")
    missing_feats = [c for c in feat_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"These feature columns are missing in dataset: {missing_feats[:10]} ...")

    X = df[feat_cols].copy()
    y = df[args.target].astype(int).copy()
    groups = df[args.group_col] if args.group_col in df.columns else None

    # Subsample for speed (group-aware)
    X_sub, y_sub, groups_sub = pick_subset(X, y, groups, frac=args.subsample_frac, seed=args.random_state)
    print(f"→ Using subset for importance: n={len(X_sub)} rows, d={X_sub.shape[1]} features")

    # Compute permutation importance for requested scores
    rows = []
    if args.score in ("ap", "both"):
        imp_ap = compute_perm_importance(pipe, X_sub, y_sub, scoring="average_precision",
                                         n_repeats=args.n_repeats, seed=args.random_state)
        imp_ap["metric"] = "average_precision"
        rows.append(imp_ap)
        barplot(imp_ap, "AP", args.topk, "reports/figures/importance_permutation_ap.png")

    if args.score in ("roc", "both"):
        imp_roc = compute_perm_importance(pipe, X_sub, y_sub, scoring="roc_auc",
                                          n_repeats=args.n_repeats, seed=args.random_state)
        imp_roc["metric"] = "roc_auc"
        rows.append(imp_roc)
        barplot(imp_roc, "ROC-AUC", args.topk, "reports/figures/importance_permutation_roc.png")

    if not rows:
        raise ValueError("No metric selected to compute permutation importance.")

    imp_all = pd.concat(rows, axis=0, ignore_index=True)
    imp_all.to_csv("reports/metrics/importance_permutation.csv", index=False)

    print("✓ Saved: reports/metrics/importance_permutation.csv")
    if args.score in ("ap", "both"):
        print("✓ Saved: reports/figures/importance_permutation_ap.png")
    if args.score in ("roc", "both"):
        print("✓ Saved: reports/figures/importance_permutation_roc.png")
    print("Done.")


if __name__ == "__main__":
    main()
