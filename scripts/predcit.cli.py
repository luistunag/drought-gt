#!/usr/bin/env python3
"""
Predict CLI for drought-gt.
- Loads calibrated model artifacts
- Applies the SAME preprocessing embedded in the saved pipeline
- Selects the expected feature columns (from columns.json)
- Outputs probabilities and binary predictions with a configurable threshold

Usage:
  python scripts/predict_cli.py \
    --model models/artifacts/model.joblib \
    --columns models/artifacts/columns.json \
    --input path/to/new_data.csv \
    --output predictions.csv \
    --threshold 0.5

Notes:
- Input must contain the numeric feature columns used during training.
- Extra columns are ignored; missing required columns will raise an error.
"""
import argparse
import json
import os
import sys
from typing import List

import joblib
import numpy as np
import pandas as pd


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict on a new CSV with a saved calibrated model.")
    p.add_argument("--model", required=True, help="Path to model.joblib")
    p.add_argument("--columns", required=True, help="Path to columns.json (feature list)")
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--output", required=True, help="Path to output CSV with predictions")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class")
    return p.parse_args(argv)


def load_feature_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cols = data.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError("Invalid columns.json: 'feature_columns' must be a non-empty list.")
    return cols


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.columns):
        raise FileNotFoundError(f"Columns file not found: {args.columns}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    print(f"→ Loading model:   {args.model}")
    model = joblib.load(args.model)

    feature_cols = load_feature_columns(args.columns)
    print(f"→ Expected features: {len(feature_cols)} columns")

    print(f"→ Reading input:   {args.input}")
    df = pd.read_csv(args.input, low_memory=False)

    # Validate columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    # Select and order columns as during training
    X = df[feature_cols].values

    # Predict probabilities and classes
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= args.threshold).astype(int)

    # Prepare output
    out = df.copy()
    out["prob_sequia"] = prob
    out[f"pred_sequia_th_{args.threshold:.2f}"] = pred

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"✓ Saved predictions: {args.output}")

    # Quick summary
    pos_rate = float(np.mean(pred))
    print(f"→ Positive rate @ {args.threshold:.2f}: {pos_rate:.3f} ({pred.sum()} / {len(pred)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
