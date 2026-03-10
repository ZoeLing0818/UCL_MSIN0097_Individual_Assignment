from pathlib import Path

SEED = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "bank_churn_dataset.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TABLES_DIR = ARTIFACTS_DIR / "tables"
AUDIT_DIR = ARTIFACTS_DIR / "audit"

for folder in [ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, TABLES_DIR, AUDIT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
