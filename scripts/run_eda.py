#!/usr/bin/env python3
"""
Run a lightweight, reproducible EDA for drought-gt.

Inputs (CSV):
  - dataset with binary target (default: sequia)
  - optional date column to enable time-aware summaries (not required)

Outputs:
  reports/metrics/eda/
    - eda_schema.csv                  : column, dtype, %NaN
    - eda_missingness.csv             : %NaN for all columns
    - eda_target_balance.csv          : class counts/ratios of target
    - eda_target_corr.csv             : correlation of each numeric feature vs target (point-biserial)
    - eda_num_corr_topk.csv           : pearson corr matrix (top-k by |corr with target|)
  reports/figures/eda/
    - target_balance.png
    - missingness_top.png
    - corr_vs_target_top.png
    - corr_heatmap_topk.png
    - hist_PRECTOTCORR.png, hist_T2M.png, ... (si existen)
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Run EDA and export basic tables/plots.")
    p.add_argument("--input", default="data/processed/dataset_target.csv", help="Input CSV path")
    p.add_argument("--target", default="sequia", help="Binary target column")
    p.add_argument("--date-col", default=None, help="Optional date column name")
    p.add_argument("--group-col", default="block_id", help="Optional group column to ignore in features")
    p.add_argument("--outdir", default="reports", help="Base output dir for reports")
    p.add_argument("--topk", type=int, default=30, help="Top-k features by |corr(target)| for heatmap/bars")
    return p.parse_args()


def ensure_dirs(base_out: str):
    figs = os.path.join(base_out, "figures", "eda")
    mets = os.path.join(base_out, "metrics", "eda")
    os.makedirs(figs, exist_ok=True)
    os.makedirs(mets, exist_ok=True)
    return figs, mets


def select_numeric_columns(df: pd.DataFrame, target: str, exclude: List[str]) -> List[str]:
    num_cols = []
    for c in df.columns:
        if c in exclude or c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


def save_schema(df: pd.DataFrame, out_csv: str):
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes.values],
        "na_frac": df.isna().mean().values
    }).sort_values("column")
    schema.to_csv(out_csv, index=False)
    return schema


def plot_target_balance(y: pd.Series, figs_dir: str, mets_dir: str, target: str):
    counts = y.value_counts().sort_index()
    ratios = (counts / counts.sum()).rename("ratio")
    out = pd.concat([counts.rename("count"), ratios], axis=1)
    out.to_csv(os.path.join(mets_dir, "eda_target_balance.csv"))

    plt.figure()
    plt.bar(out.index.astype(str), out["count"].values)
    plt.title(f"Target balance: {target}")
    plt.xlabel(target)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "target_balance.png"), dpi=160)
    plt.close()


def plot_missingness(df: pd.DataFrame, figs_dir: str, mets_dir: str, topk: int = 30):
    miss = df.isna().mean().sort_values(ascending=False)
    miss.to_csv(os.path.join(mets_dir, "eda_missingness.csv"), header=["na_frac"])

    top = miss.head(topk)
    plt.figure()
    plt.barh(range(len(top)), top.values)
    plt.yticks(range(len(top)), top.index)
    plt.gca().invert_yaxis()
    plt.xlabel("NaN fraction")
    plt.title(f"Missingness (top {len(top)})")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "missingness_top.png"), dpi=160)
    plt.close()


def corr_vs_target(df: pd.DataFrame, num_cols: List[str], y: pd.Series, mets_dir: str, figs_dir: str, topk: int):
    # Point-biserial = Pearson between numeric and {0,1}
    y_float = y.astype(float)
    corrs = []
    for c in num_cols:
        s = df[c]
        if s.notna().sum() < 3:
            r = np.nan
        else:
            r = pd.Series(s).corr(y_float)  # handles NaNs pairwise
        corrs.append((c, r))
    corr_df = pd.DataFrame(corrs, columns=["feature", "corr_target"]).sort_values(
        "corr_target", key=lambda s: s.abs(), ascending=False
    )
    corr_df.to_csv(os.path.join(mets_dir, "eda_target_corr.csv"), index=False)

    top = corr_df.head(topk)
    plt.figure(figsize=(8, max(4, 0.3 * len(top))))
    plt.barh(range(len(top)), top["corr_target"].values)
    plt.yticks(range(len(top)), top["feature"].values)
    plt.gca().invert_yaxis()
    plt.xlabel("Pearson corr with target")
    plt.title(f"Top {len(top)} |corr| vs target")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "corr_vs_target_top.png"), dpi=160)
    plt.close()

    return corr_df


def corr_heatmap_topk(df: pd.DataFrame, top_features: List[str], figs_dir: str, mets_dir: str):
    # compute pairwise Pearson corr on selected features (drop all-NaN columns)
    sub = df[top_features].copy()
    # Optional: impute a tiny bit for heatmap to avoid empty rows/cols
    sub = sub.fillna(sub.median(numeric_only=True))
    corr = sub.corr(method="pearson")
    corr.to_csv(os.path.join(mets_dir, "eda_num_corr_topk.csv"))

    # basic heatmap with matplotlib
    plt.figure(figsize=(max(6, 0.35 * len(top_features)), max(5, 0.35 * len(top_features))))
    im = plt.imshow(corr.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Pearson correlation (top-k by |corr vs target|)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "corr_heatmap_topk.png"), dpi=160)
    plt.close()


def quick_hists(df: pd.DataFrame, figs_dir: str):
    # Optional “usual suspects” if present
    candidates = ["PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX", "RH2M"]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if s.empty:
                continue
            plt.figure()
            plt.hist(s.values, bins=40)
            plt.title(f"Histogram: {c}")
            plt.xlabel(c)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"hist_{c}.png"), dpi=160)
            plt.close()


def main():
    args = parse_args()
    figs_dir, mets_dir = ensure_dirs(args.outdir)

    df = pd.read_csv(args.input, low_memory=False)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in {args.input}")

    # Save schema / missingness
    save_schema(df, os.path.join(mets_dir, "eda_schema.csv"))
    plot_missingness(df, figs_dir, mets_dir, topk=args.topk)

    # Target balance
    y = df[args.target]
    if set(pd.unique(y)) - {0, 1}:
        raise ValueError(f"Target '{args.target}' must be binary (0/1).")
    plot_target_balance(y, figs_dir, mets_dir, args.target)

    # Numeric columns (exclude id/coords/date-like/group columns)
    exclude = [args.target, args.group_col, "LAT", "LON", "YEAR", "MONTH"]
    if args.date_col and args.date_col in df.columns:
        exclude.append(args.date_col)
    num_cols = select_numeric_columns(df, args.target, exclude)

    if not num_cols:
        raise ValueError("No numeric feature columns found for EDA.")

    # Corr with target (point-biserial) and top-k selection
    corr_df = corr_vs_target(df, num_cols, y, mets_dir, figs_dir, topk=args.topk)
    top_features = corr_df["feature"].head(args.topk).tolist()

    # Heatmap on top-k features
    corr_heatmap_topk(df, top_features, figs_dir, mets_dir)

    # Quick histograms for key variables if present
    quick_hists(df, figs_dir)

    print("✓ EDA tables saved to:", mets_dir)
    print("✓ EDA figures saved to:", figs_dir)


if __name__ == "__main__":
    main()
