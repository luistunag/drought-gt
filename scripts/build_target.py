#!/usr/bin/env python3
"""
Build binary drought target from monthly precipitation climatology.

- Groups by (LAT, LON, MONTH) and computes a percentile of PRECTOTCORR
- Labels sequia=1 if current PRECTOTCORR <= climatology_percentile
- Writes a new CSV with a 'sequia' column

Usage (defaults):
  python scripts/build_target.py \
    --input data/processed/dataset_enriched.csv \
    --output data/processed/dataset_target.csv \
    --precip-col PRECTOTCORR \
    --lat-col LAT --lon-col LON \
    --month-col MONTH \
    --percentile 20
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build drought target from monthly precipitation climatology.")
    p.add_argument("--input", "-i", default="data/processed/dataset_enriched.csv")
    p.add_argument("--output", "-o", default="data/processed/dataset_target.csv")
    p.add_argument("--precip-col", default="PRECTOTCORR")
    p.add_argument("--lat-col", default="LAT")
    p.add_argument("--lon-col", default="LON")
    # Either provide MONTH directly, or pass --date-col and the script will extract month
    p.add_argument("--month-col", default="MONTH")
    p.add_argument("--date-col", default=None)
    p.add_argument("--percentile", type=float, default=20.0, help="Drought threshold percentile (0-100).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    needed = [args.precip_col, args.lat_col, args.lon_col]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in dataset: {args.input}")

    # Resolve month column
    if args.month_col and args.month_col in df.columns:
        month = df[args.month_col].astype(int)
    elif args.date_col and args.date_col in df.columns:
        month = pd.to_datetime(df[args.date_col]).dt.month
    else:
        raise ValueError("Provide --month-col (present in data) or --date-col to extract month.")

    df["_MONTH_RESOLVED_"] = month.values

    # Compute climatology percentile by (LAT, LON, MONTH)
    group_keys = [args.lat_col, args.lon_col, "_MONTH_RESOLVED_"]
    if df[args.precip_col].isna().all():
        raise ValueError(f"'{args.precip_col}' has only NaNs.")

    def pct_func(x):
        return np.nanpercentile(x, args.percentile)

    clim = (
        df.groupby(group_keys, dropna=False)[args.precip_col]
        .apply(pct_func)
        .reset_index()
        .rename(columns={args.precip_col: "ppt_pctl"})
    )

    out = df.merge(clim, on=group_keys, how="left")
    # sequia = 1 if current precip <= percentile threshold
    out["sequia"] = (out[args.precip_col] <= out["ppt_pctl"]).astype(int)

    # Basic stats
    prev = float(out["sequia"].mean())
    n = len(out)
    print(f"→ Drought label built @ p{args.percentile:.1f}: prevalence={prev:.3f} (positives={out['sequia'].sum()}/{n})")

    # Cleanup helper column
    out = out.drop(columns=["_MONTH_RESOLVED_"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"✓ Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
