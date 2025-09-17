#!/usr/bin/env python3
"""
Add a spatial block_id column to a dataset for spatial CV.

- Reads a CSV with LAT, LON columns
- Assigns each row to a spatial block (~km-sized grid cells)
- Writes an output CSV (default: data/processed/dataset_blocks.csv)
- Then you can run train/evaluate with: --group-col block_id

Usage:
  python scripts/add_spatial_blocks.py \
      --input data/processed/dataset.csv \
      --output data/processed/dataset_blocks.csv \
      --block-km 50

Requires: pandas, numpy
"""
import argparse
import math
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def km_to_deg_lat(km: float) -> float:
    # ~111.32 km per degree of latitude
    return km / 111.32


def km_to_deg_lon(km: float, lat_deg: float) -> float:
    # ~111.32 * cos(latitude) km per degree of longitude
    return km / (111.32 * max(1e-6, math.cos(math.radians(lat_deg))))


def compute_block_indices(lat: np.ndarray, lon: np.ndarray, block_km: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (lat, lon) to integer grid indices using ~block_km sized cells.
    Longitude degree width varies with latitude; we compute per-row.
    """
    # Latitude degree size is constant
    dlat = km_to_deg_lat(block_km)

    # Per-row longitude degree size (depends on latitude)
    dlon_arr = np.array([km_to_deg_lon(block_km, la) for la in lat], dtype=float)

    # Shift to positive by adding large offsets (so floor is stable)
    lat_idx = np.floor((lat + 90.0) / dlat).astype(np.int64)

    # For lon, use per-row dlon
    lon_idx = np.floor((lon + 180.0) / dlon_arr).astype(np.int64)
    return lat_idx, lon_idx


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Add spatial block_id column for spatial CV.")
    p.add_argument("--input", "-i", default="data/processed/dataset.csv", help="Input CSV path.")
    p.add_argument("--output", "-o", default="data/processed/dataset_blocks.csv", help="Output CSV path.")
    p.add_argument("--block-km", type=float, default=50.0, help="Approximate block size in kilometers.")
    args = p.parse_args(argv)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    for col in ("LAT", "LON"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)[:20]} ...")

    lat = df["LAT"].astype(float).values
    lon = df["LON"].astype(float).values

    lat_idx, lon_idx = compute_block_indices(lat, lon, args.block_km)
    df["block_id"] = (lat_idx.astype(str) + "_" + lon_idx.astype(str))

    # Optional: make block_id categorical for smaller file size
    df["block_id"] = df["block_id"].astype("category")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"✓ Added 'block_id' using ~{args.block_km:.1f} km cells")
    print(f"→ Saved: {args.output}")
    print("Next: use --group-col block_id in train/evaluate scripts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
