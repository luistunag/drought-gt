#!/usr/bin/env python3
"""
Evaluate a saved calibrated model on a HOLDOUT CSV.

Outputs:
  - reports/metrics/holdout_metrics.json
  - reports/figures/pr_curve_holdout.png
  - reports/figures/roc_curve_holdout.png
  - reports/figures/reliability_curve_holdout.png
  - reports/figures/confusion_matrix_holdout.png
"""
import argparse
import json
import os
import sys
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    brier_score_loss,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Holdout evaluation for drought-gt.")
    p.add_argument("--model", default="models/artifacts/model.joblib")
    p.add_argument("--columns", default="models/artifacts/columns.json")
    p.add_argument("--input", required=True, help="Path to HOLDOUT CSV")
    p.add_argument("--target", default="sequia")
    p.add_argument("--outdir", default="reports")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args(argv)


def load_feature_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    cols = obj.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError("Invalid columns.json: 'feature_columns' must be a non-empty list.")
    return cols


def ensure_dirs(base: str) -> Dict[str, str]:
    figs = os.path.join(base, "figures")
    mets = os.path.join(base, "metrics")
    os.makedirs(figs, exist_ok=True)
    os.makedirs(mets, exist_ok=True)
    return {"figs": figs, "mets": mets}


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = ensure_dirs(args.outdir)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.columns):
        raise FileNotFoundError(f"Columns file not found: {args.columns}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Holdout CSV not found: {args.input}")

    print(f"→ Loading model:   {args.model}")
    model = joblib.load(args.model)
    feature_cols = load_feature_columns(args.columns)

    print(f"→ Reading holdout: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in holdout CSV.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Holdout missing required feature columns: {missing}")

    X = df[feature_cols].values
    y = df[args.target].values
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Target must be binary (0/1).")

    # Predictions
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= args.threshold).astype(int)

    # Metrics
    pr_auc = average_precision_score(y, prob)
    roc_auc = roc_auc_score(y, prob)
    f1 = f1_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    brier = brier_score_loss(y, prob)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    metrics = {
        "threshold": args.threshold,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "recall": rec,
        "precision": prec,
        "brier": brier,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y)),
        "positive_rate_pred": float(np.mean(pred)),
        "positive_rate_true": float(np.mean(y)),
    }

    # Save metrics JSON
    metrics_path = os.path.join(paths["mets"], "holdout_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved: {metrics_path}")

    # --- Plots ---
    # PR curve
    p, r, _ = precision_recall_curve(y, prob)
    plt.figure()
    plt.step(r, p, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Holdout PR curve (AP={pr_auc:.3f})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    pr_path = os.path.join(paths["figs"], "pr_curve_holdout.png")
    plt.savefig(pr_path, dpi=160)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Holdout ROC curve (AUC={roc_auc:.3f})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    roc_path = os.path.join(paths["figs"], "roc_curve_holdout.png")
    plt.savefig(roc_path, dpi=160)
    plt.close()

    # Reliability curve (calibration)
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.7)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Holdout reliability (Brier={brier:.3f})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    rel_path = os.path.join(paths["figs"], "reliability_curve_holdout.png")
    plt.savefig(rel_path, dpi=160)
    plt.close()

    # Confusion matrix plot (simple)
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix @ th={args.threshold:.2f}")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    cm_path = os.path.join(paths["figs"], "confusion_matrix_holdout.png")
    plt.savefig(cm_path, dpi=160)
    plt.close()

    print(f"Saved figures → {paths['figs']}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
