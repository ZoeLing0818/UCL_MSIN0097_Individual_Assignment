# MSIN0097 Predictive Analytics - Individual Assignment (VWNP8)

## 1. Project Overview
This project predicts bank customer attrition (`Attrition_Flag`) using a reproducible ML pipeline.
The workflow is split into modular scripts for data preparation, model comparison, tuning, and evaluation.

## 2. Repository Structure
```text
.
├── data/
│   └── bank_churn_dataset.csv
├── notebooks/
│   ├── MSIN0097_VWNP8.ipynb
│   └── MSIN0097_VWNP8_industrial.ipynb
├── src/
│   ├── data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── run_all.py
├── artifacts/
│   ├── models/
│   ├── plots/
│   ├── reports/
│   └── tables/
└── requirements.txt
```

## 3. Environment Setup
- Python: `3.12.x` (recommended)
- Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## 4. How to Run
Run step-by-step:

```bash
python3 -m src.train
python3 -m src.tune
python3 -m src.evaluate
```

Or run all in one:

```bash
python3 -m src.run_all
```

## 5. Notebook Usage
- Main write-up notebook: `notebooks/MSIN0097_VWNP8_industrial.ipynb`
- The notebook reads outputs generated under `artifacts/` for reporting and discussion.

## 6. Output Artifacts
After running the pipeline, key outputs include:
- Model comparison table: `artifacts/tables/step4_model_comparison.csv`
- Tuning report: `artifacts/reports/step5_tuning_report.json`
- Evaluation report: `artifacts/reports/step5_evaluation_report.json`
- Saved model: `artifacts/models/step5_best_model.pkl`
- Plots: `artifacts/plots/`

## 7. Reproducibility
- Global random seed is fixed in `src/settings.py` (`SEED = 42`).
- Train/test split uses stratified sampling for class balance.

## 8. Notes / Limitations
- Dataset is static tabular data; temporal drift is not explicitly modeled.
- Threshold and calibration may need re-tuning if business costs change.