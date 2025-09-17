# scripts/holdout_summary.py
import json, sys
if len(sys.argv) < 2:
    print("usage: python scripts/holdout_summary.py <metrics_json>")
    sys.exit(2)
path = sys.argv[1]
with open(path) as f:
    m = json.load(f)
print(f"PR-AUC     : {m['pr_auc']:.4f}")
print(f"ROC-AUC    : {m['roc_auc']:.4f}")
print(f"F1         : {m['f1']:.4f}")
print(f"Recall     : {m['recall']:.4f}")
print(f"Precision  : {m['precision']:.4f}")
print(f"Brier      : {m['brier']:.4f}")
