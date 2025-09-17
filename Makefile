# ---- config (override on CLI: make train TARGET=sequia) ----
PYTHON        ?= python
CONFIG        ?= config/default.yaml
TARGET        ?= sequia
DATASET       ?= data/processed/dataset.csv
DATASET_BLK   ?= data/processed/dataset_blocks.csv
DATASET_FE    ?= data/processed/dataset_enriched.csv
DATASET_SEL   ?= data/processed/dataset_selected.csv
DATASET_TGT   ?= data/processed/dataset_target.csv   
DATASET_TRAIN ?= $(DATASET_TGT) 
GROUP_COL     ?= block_id
BLOCK_KM      ?= 50
DATE_COL      ?=
FE_COLS       ?=        # e.g., "PRECTOTCORR T2M T2M_MIN T2M_MAX"
LAGS          ?= 1 2 3
WINDOWS       ?= 3 6


# ---- phony ----
.PHONY: data blocks features target train eval smote calibrate fit predict help holdout

help:
	@echo "make data        # build processed dataset (merge climate+soil)"
	@echo "make blocks      # add spatial blocks (block_id) to dataset"
	@echo "make features    # add anomalies/lags/rolling features (uses DATE_COL if provided)"
	@echo "make target      # build binary drought label from precip climatology (sets 'sequia')"
	@echo "make train       # baseline CV training (uses dataset_target.csv)"
	@echo "make eval        # baseline CV evaluation + PR/ROC plots"
	@echo "make smote       # CV training with imbalance handling (SMOTE/class_weight)"
	@echo "make calibrate   # CV training with calibration (sigmoid/isotonic)"
	@echo "make fit         # fit full calibrated model and save artifacts"
	@echo "make predict     # run predictions on a new CSV"
	@echo "make holdout     # evaluate saved model on a holdout CSV"
	@echo ""
	@echo "Override vars: TARGET, DATASET, DATASET_BLK, DATASET_FE, DATASET_TGT, GROUP_COL, BLOCK_KM, DATE_COL, FE_COLS, LAGS, WINDOWS, PERCENTILE"
	@echo "Examples:"
	@echo "  make features DATE_COL=date FE_COLS='PRECTOTCORR T2M T2M_MIN T2M_MAX' LAGS='1 2 3' WINDOWS='3 6'"
	@echo "  make target PERCENTILE=20"

# ---- steps ----

data:
	$(PYTHON) scripts/make_dataset.py --config $(CONFIG)

blocks:
	$(PYTHON) scripts/add_spatial_blocks.py \
	  --input $(DATASET) \
	  --output $(DATASET_BLK) \
	  --block-km $(BLOCK_KM)

eda:
	$(PYTHON) scripts/run_eda.py \
	  --input $(DATASET_TGT) \
	  --target $(TARGET) \
	  $(if $(strip $(DATE_COL)),--date-col $(DATE_COL),) \
	  --group-col $(GROUP_COL) \
	  --outdir reports \
	  --topk 30

# If DATE_COL empty -> anomalies only; if set -> anomalies + lags + rolling
features:
	$(PYTHON) scripts/features_add_basic.py \
	  --input $(DATASET_BLK) \
	  --output $(DATASET_FE) \
	  --target $(TARGET) \
	  $(if $(strip $(DATE_COL)),--date-col $(DATE_COL),) \
	  $(if $(strip $(FE_COLS)),--cols $(FE_COLS),) \
	  --lags $(LAGS) \
	  --windows $(WINDOWS)

# Build binary drought label from precipitation climatology
# Usage: make target PERCENTILE=20
target:
	$(if $(PERCENTILE),,$(eval PERCENTILE=20))
	$(PYTHON) scripts/build_target.py \
	  --input $(DATASET_FE) \
	  --output $(DATASET_TGT) \
	  --precip-col PRECTOTCORR \
	  --lat-col LAT --lon-col LON \
	  --month-col MONTH \
	  --percentile $(PERCENTILE)

select_features:
	$(PYTHON) scripts/select_features.py \
	  --input $(DATASET_TGT) \
	  --out $(DATASET_SEL) \
	  --target $(TARGET) \
	  --group-col $(GROUP_COL) \
	  --method rf \
	  --n-splits 5 \
	  --topk 50 \
	  --min-importance 0.0

train:
	$(PYTHON) scripts/train_baseline.py \
	  --dataset $(DATASET_TRAIN) \
	  --target $(TARGET) \
	  --group-col $(GROUP_COL)

eval:
	$(PYTHON) scripts/evaluate_baseline.py \
	  --dataset $(DATASET_TRAIN) \
	  --target $(TARGET) \
	  --group-col $(GROUP_COL)

# sampler options: smote | class_weight | none
smote:
	$(PYTHON) scripts/train_smote.py \
	  --dataset $(DATASET_TRAIN) \
	  --target $(TARGET) \
	  --group-col $(GROUP_COL) \
	  --sampler smote

# calibration options: sigmoid | isotonic
calibrate:
	$(PYTHON) scripts/train_calibrated.py \
	  --dataset $(DATASET_TRAIN) \
	  --target $(TARGET) \
	  --group-col $(GROUP_COL) \
	  --sampler smote \
	  --calibration isotonic \
	  --inner-cv 3

fit:
	$(PYTHON) scripts/fit_full_model.py \
	  --dataset $(DATASET_TRAIN) \
	  --target $(TARGET) \
	  --group-col $(GROUP_COL) \
	  --sampler smote \
	  --calibration isotonic \
	  --inner-cv 3

# Example:
# make predict INPUT=data/new_samples.csv OUT=reports/preds/new_with_preds.csv TH=0.5
predict:
	$(if $(INPUT),,$(error Set INPUT=path/to.csv))
	$(if $(OUT),,$(error Set OUT=path/to_output.csv))
	$(if $(TH),,$(eval TH=0.5))
	$(PYTHON) scripts/predict_cli.py \
	  --model models/artifacts/model.joblib \
	  --columns models/artifacts/columns.json \
	  --input $(INPUT) \
	  --output $(OUT) \
	  --threshold $(TH)

# Example:
# make holdout INPUT=data/processed/holdout.csv TH=0.5 OUTDIR=reports
holdout:
	$(if $(INPUT),,$(error Set INPUT=path/to_holdout.csv))
	$(if $(TH),,$(eval TH=0.5))
	$(if $(OUTDIR),,$(eval OUTDIR=reports))
	$(PYTHON) scripts/eval_holdout.py \
	  --model models/artifacts/model.joblib \
	  --columns models/artifacts/columns.json \
	  --input $(INPUT) \
	  --target $(TARGET) \
	  --outdir $(OUTDIR) \
	  --threshold $(TH)
	@echo ""
	@echo "=== Holdout metrics (full JSON) ==="
	@cat $(OUTDIR)/metrics/holdout_metrics.json
	@echo ""
	@echo "=== Holdout metrics (summary) ==="
	@$(PYTHON) scripts/holdout_summary.py $(OUTDIR)/metrics/holdout_metrics.json

importance:
	$(PYTHON) scripts/analyze_importance.py \
	  --dataset $(DATASET_TRAIN) \
	  --target $(TARGET) \
	  --group-col $(GROUP_COL) \
	  --model models/artifacts/model.joblib \
	  --columns models/artifacts/columns.json \
	  --score both \
	  --subsample-frac 0.35 \
	  --n-repeats 10 \
	  --topk 40
