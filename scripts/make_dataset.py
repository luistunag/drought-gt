#!/usr/bin/env python3
"""
Builds a processed dataset by merging climate and soil files.

Usage:
  python scripts/make_dataset.py --config config/default.yaml

Requires: pandas, pyyaml
"""
import argparse
import os
import sys
from typing import List, Dict, Any

import pandas as pd
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def validate_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in '{name}': {missing}. "
            f"Available: {list(df.columns)}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build processed dataset (merge climate + soil)."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/default.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = cfg["paths"]["raw"]
    processed_dir = cfg["paths"]["processed"]
    ensure_dir(processed_dir)

    climate_path = os.path.join(raw_dir, cfg["data"]["climate_file"])
    soil_path = os.path.join(raw_dir, cfg["data"]["soil_file"])
    output_path = os.path.join(processed_dir, cfg["data"]["output_file"])
    merge_keys = cfg["data"]["merge_keys"]

    print(f"→ Reading climate: {climate_path}")
    climate = pd.read_csv(climate_path, low_memory=False)
    print(f"   rows={len(climate):,}  cols={len(climate.columns)}")

    print(f"→ Reading soil:    {soil_path}")
    soil = pd.read_csv(soil_path, low_memory=False)
    print(f"   rows={len(soil):,}  cols={len(soil.columns)}")

    # Minimal validations
    validate_columns(climate, cfg["validate"]["required_climate_cols"], "climate")
    validate_columns(soil, cfg["validate"]["required_soil_cols"], "soil")
    for k in merge_keys:
        if k not in climate.columns or k not in soil.columns:
            raise ValueError(f"Merge key '{k}' must exist in both files.")

    # Merge (left join on climate)
    print(f"→ Merging on keys {merge_keys} (left join on climate)…")
    merged = pd.merge(climate, soil, on=merge_keys, how="left")

    # Quick report
    null_pct = merged.isna().mean().mean()
    print(
        f"✓ Merge completed: rows={len(merged):,}  cols={len(merged.columns)}  "
        f"avg_%NaN={null_pct:.2%}"
    )

    # Add proper monthly date column from YEAR + MONTH
    if "YEAR" in merged.columns and "MONTH" in merged.columns:
        merged["date"] = pd.to_datetime(
            merged["YEAR"].astype(str) + "-" + merged["MONTH"].astype(str) + "-01"
        )
    
    # Save
    merged.to_csv(output_path, index=False)
    print(f"💾 Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
