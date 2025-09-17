# drought-gt

Minimal, reproducible pipeline step 1: build a processed dataset by merging climate and soil files.

# Project layout (current step)
```
drought-gt/
├─ config/
│  └─ default.yaml
├─ scripts/
│  └─ make_dataset.py
├─ data/
│  ├─ raw/         # place source CSVs here
│  └─ processed/   # output dataset.csv is written here
└─ .gitignore
```

Requirements

	Python 3.10+

	pandas, pyyaml

Install:

```
pip install pandas pyyaml
```

# Usage

	1. Put your input files in data/raw/ and ensure the names in config/default.yaml match:

	- monthly_climate.csv

	- soil_data.csv

	2. Run the dataset builder:

```
python scripts/make_dataset.py --config config/default.yaml
```

	3. The merged dataset is saved to:

```
data/processed/dataset.csv
```

# Notes

	Merge keys are ["LAT", "LON"] by default; change them in config/default.yaml if needed.

	The script prints a quick summary (rows, columns, average %NaN).
