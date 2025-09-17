#!/usr/bin/env python3
"""
Model-based feature selection for drought-gt.

- Fits a model (RF or L1-logistic) across K folds (GroupKFold if group provided)
- Averages feature importances / absolute coefficients
- Selects features by top-k and/or min-importance threshold
- Saves:
    * reports/metrics/feat_importance_<method>.csv
    * reports/figures/feat_importance_<method>.png
    * models/selected_features.txt
    * data/processed/dataset_selected.csv  (filtered columns + target + ids)

Usage (example):
  python scripts/select_features.py \
    --input data/processed/dataset_target.csv \
    --target sequia \
    --group-col block_id \
    --out data/processed/dataset_selected.csv \
    --method rf \
    --n-splits 5 \
    --topk 50 \
    --min-importance 0.0
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def parse_args():
    p = argparse.ArgumentParser(description="Model-based feature selection.")
    p.add_argument("--input", default="data/processed/dataset_target.csv")
    p.add_argument("--out", default="data/processed/dataset_selected.csv")
    p.add_argument("--target", default="sequia")
    p.add_argument("--group-col", default="block_id")
    p.add_argument("--method", choices=["rf", "l1"], default="rf")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--topk", type=int, default=50, help="Keep top-k features by importance (0 = keep all)")
    p.add_argument("--min-importance", type=float, default=0.0, help="Keep features with importance >= this")
    # RF params
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=None)
    # L1 params
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


def ensure_dirs():
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def select_numeric_columns(df: pd.DataFrame, target: str, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude or c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_preprocess_for(method: str, num_cols: List[str]) -> ColumnTransformer:
    """
    RF: imputación mediana (no es necesario escalar)
    L1: imputación + escalado
    """
    if method == "rf":
        num_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
    else:  # l1
        num_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model(method: str, args) -> object:
    if method == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=-1,
            class_weight="balanced",
            random_state=args.random_state,
        )
    else:  # l1
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=args.C,
            class_weight="balanced",
            random_state=args.random_state,
            max_iter=2000,
        )


def fold_importances(estimator, method: str, feature_names: List[str]) -> np.ndarray:
    if method == "rf":
        imp = np.asarray(estimator.feature_importances_, dtype=float)
    else:  # l1
        coef = estimator.coef_.ravel()
        imp = np.abs(coef)
    # Align length
    if len(imp) != len(feature_names):
        raise RuntimeError("Importer length != #features after preprocessing.")
    return imp


def main():
    args = parse_args()
    ensure_dirs()

    df = pd.read_csv(args.input, low_memory=False)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in {args.input}")

    # Exclude ids/date/group coords from features
    exclude = [args.target, args.group_col, "LAT", "LON", "YEAR", "MONTH", "date"]
    num_cols = select_numeric_columns(df, args.target, exclude)
    if not num_cols:
        raise ValueError("No numeric features detected for selection.")

    # Exclude obvious leakage features (current-month precip and its direct transforms)
    leakage = {
        "PRECTOTCORR", "ppt_pctl",
        "PRECTOTCORR__z", "PRECTOTCORR__roll3m", "PRECTOTCORR__roll6m",
    }
    num_cols = [c for c in num_cols if c not in leakage]

    X = df[num_cols]            # mantener DataFrame para ColumnTransformer
    y = df[args.target].values
    groups = df[args.group_col].values if args.group_col in df.columns else None

    splitter = GroupKFold(n_splits=args.n_splits) if groups is not None else KFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.random_state
    )

    # Acumular importancias por fold
    imps = []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_tr = X.iloc[tr]
        y_tr = y[tr]
        pre = make_preprocess_for(args.method, num_cols)
        clf = make_model(args.method, args)
        pipe = SkPipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)

        # extraer importancias después del preprocesado
        est = pipe.named_steps["clf"]
        imp = fold_importances(est, args.method, num_cols)
        imps.append(imp)

    imps = np.vstack(imps)
    mean_imp = imps.mean(axis=0)
    std_imp = imps.std(axis=0)

    imp_df = pd.DataFrame({
        "feature": num_cols,
        "importance_mean": mean_imp,
        "importance_std": std_imp
    }).sort_values("importance_mean", ascending=False)

    # Guardar importancias
    method_tag = args.method
    imp_csv = f"reports/metrics/feat_importance_{method_tag}.csv"
    imp_df.to_csv(imp_csv, index=False)

    # Plot top 40
    top_plot = imp_df.head(40)
    plt.figure(figsize=(9, max(4, 0.3 * len(top_plot))))
    plt.barh(range(len(top_plot)), top_plot["importance_mean"].values, xerr=top_plot["importance_std"].values)
    plt.yticks(range(len(top_plot)), top_plot["feature"].values)
    plt.gca().invert_yaxis()
    plt.xlabel("importance (mean ± std)")
    plt.title(f"Feature importance ({method_tag})")
    plt.tight_layout()
    plt.savefig(f"reports/figures/feat_importance_{method_tag}.png", dpi=160)
    plt.close()

    # Selección
    selected = imp_df.copy()
    if args.min_importance > 0:
        selected = selected[selected["importance_mean"] >= args.min_importance]
    if args.topk and args.topk > 0:
        selected = selected.head(args.topk)

    selected_features = selected["feature"].tolist()
    if not selected_features:
        # fallback: mantén al menos las top 20
        selected_features = imp_df.head(20)["feature"].tolist()

    # Guardar lista
    with open("models/selected_features.txt", "w", encoding="utf-8") as f:
        for c in selected_features:
            f.write(c + "\n")

    # Crear dataset filtrado (mantén ids y target)
    keep_cols = selected_features + [args.target]
    for extra in ["LAT", "LON", "YEAR", "MONTH", "date", args.group_col]:
        if extra in df.columns and extra not in keep_cols:
            keep_cols.append(extra)
    sel_df = df[keep_cols].copy()
    sel_df.to_csv(args.out, index=False)

    print(f"✓ Saved importance table: {imp_csv}")
    print(f"✓ Saved plot: reports/figures/feat_importance_{method_tag}.png")
    print(f"✓ Saved selected feature names: models/selected_features.txt (n={len(selected_features)})")
    print(f"✓ Saved filtered dataset: {args.out} (cols={len(sel_df.columns)})")


if __name__ == "__main__":
    main()
