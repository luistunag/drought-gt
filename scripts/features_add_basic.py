#!/usr/bin/env python3
"""
Add basic features: per-group z-score anomalies, lags, rolling means, and past-only rolling from lags.

- Works with monthly data if you provide --date-col (e.g., 'date')
- Groups by keys (default: ['LAT', 'LON']) to compute features per location
- You can specify which numeric columns to transform; by default it auto-detects
  numeric climate columns (excluding target, ids, and group/date columns)

Outputs a new CSV with added columns:
- <col>__z               : z-score anomaly within group
- <col>__lag{k}          : lagged value by k steps (if date provided)
- <col>__roll{w}m        : rolling mean over w steps (if date provided; includes month t)
- <col>__roll_prev{K}m   : past-only rolling from lags (mean of lag1..lagK)  ← NO leakage

Usage examples:
  python scripts/features_add_basic.py \
      --input data/processed/dataset_blocks.csv \
      --output data/processed/dataset_enriched.csv \
      --target sequia \
      --date-col date \
      --group-keys LAT LON \
      --cols PRECTOTCORR T2M T2MDEW RH2M T2M_MAX QV2M T2M_MIN TS T2MWET PS T2M_RANGE \
      --lags 1 2 3 4 5 6 \
      --windows 3 6

Requires: pandas, numpy
"""
import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add basic anomalies/lags/rolling features.")
    p.add_argument("--input", "-i", required=True, help="Input CSV path.")
    p.add_argument("--output", "-o", required=True, help="Output CSV path.")
    p.add_argument("--target", "-t", default="sequia", help="Target column (excluded from features).")
    p.add_argument("--date-col", default=None, help="Optional date column name (monthly).")
    p.add_argument("--group-keys", nargs="*", default=["LAT", "LON"], help="Group keys for per-location features.")
    p.add_argument(
        "--cols",
        nargs="*",
        default=None,
        help="Explicit numeric columns to transform (default: auto-detect).",
    )
    p.add_argument("--lags", nargs="*", type=int, default=[1, 2, 3], help="Lag steps (requires --date-col).")
    p.add_argument("--windows", nargs="*", type=int, default=[3, 6], help="Rolling windows in steps (requires --date-col).")
    return p.parse_args(argv)


def auto_numeric_columns(
    df: pd.DataFrame, target: str, group_keys: List[str], date_col: Optional[str]
) -> List[str]:
    exclude = set([target] + group_keys)
    if date_col:
        exclude.add(date_col)
    # Common geo id fields often excluded from modeling:
    for c in ["LAT", "LON", "block_id"]:
        if c in df.columns:
            exclude.add(c)
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric candidate columns found for feature engineering.")
    return num_cols


def zscore_groupwise(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - mu) / sd


# === Add past-only rolling from lags (no leakage) ============================
# For each feature col `c` in the selected climate columns, if the lag columns
# exist in the *output* dataframe, we build:
#   c__roll_prev3m = mean(c__lag1, c__lag2, c__lag3)
#   c__roll_prev6m = mean(c__lag1, ..., c__lag6)
def add_prev_rolls(out_df: pd.DataFrame, cols: List[str], ks=(3, 6)) -> pd.DataFrame:
    for c in cols:
        for k in ks:
            lag_cols = [f"{c}__lag{i}" for i in range(1, k + 1)]
            if all(lc in out_df.columns for lc in lag_cols):
                out_df[f"{c}__roll_prev{k}m"] = out_df[lag_cols].mean(axis=1)
    return out_df
# ============================================================================


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"→ Loading: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)

    for g in args.group_keys:
        if g not in df.columns:
            raise ValueError(f"Group key '{g}' not found in dataset.")

    if args.date_col and args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found in dataset.")

    # If date column is provided, parse to datetime and sort per group
    if args.date_col:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
        if df[args.date_col].isna().any():
            raise ValueError(f"Some values in '{args.date_col}' could not be parsed as dates.")
        df = df.sort_values(args.group_keys + [args.date_col]).reset_index(drop=True)

    # Decide which columns to transform
    if args.cols:
        for c in args.cols:
            if c not in df.columns:
                raise ValueError(f"Requested column '{c}' not in dataset.")
        cols = args.cols
    else:
        cols = auto_numeric_columns(df, args.target, args.group_keys, args.date_col)

    print(f"→ Feature columns: {cols}")

    # Per-group z-score anomalies
    print("→ Computing per-group z-score anomalies …")
    df_anom = (
        df[args.group_keys + cols]
        .groupby(args.group_keys, group_keys=False)
        .transform(zscore_groupwise)
    )
    df_anom = df_anom.add_suffix("__z")
    out = pd.concat([df, df_anom], axis=1)

    # Temporal features if date is provided
    if args.date_col:
        print(f"→ Computing lags {args.lags} and rolling means {args.windows} (by {args.group_keys}) …")
        gb = out.groupby(args.group_keys, group_keys=False)
        # Lags
        lag_frames = []
        for k in args.lags:
            lag_df = gb[cols].shift(k)
            lag_df = lag_df.add_suffix(f"__lag{k}")
            lag_frames.append(lag_df)
        out = pd.concat([out] + lag_frames, axis=1)
        # Rolling means
        roll_frames = []
        for w in args.windows:
            roll_df = gb[cols].rolling(window=w, min_periods=max(1, w)).mean()
            roll_df = roll_df.reset_index(level=args.group_keys, drop=True)
            roll_df = roll_df.add_suffix(f"__roll{w}m")
            roll_frames.append(roll_df)
        out = pd.concat([out] + roll_frames, axis=1)
        # Past-only rolling built from lags (NO leakage)
        out = add_prev_rolls(out, cols, ks=(3, 6))

    # Done
    out.to_csv(args.output, index=False)
    print(f"✓ Saved with features: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
