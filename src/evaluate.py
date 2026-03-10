from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from .data import prepare_dataset
from .metrics import metric_pack
from .settings import MODELS_DIR, PLOTS_DIR, REPORTS_DIR
from .tune import run_tune_stage
from .utils import write_json


def _load_or_create_tuned_payload() -> dict:
    p = MODELS_DIR / "step5_best_model.pkl"
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    run_tune_stage()
    with p.open("rb") as f:
        return pickle.load(f)


def run_evaluate_stage() -> dict:
    ds = prepare_dataset()
    tuned_payload = _load_or_create_tuned_payload()

    best_model = tuned_payload["model"]
    best_threshold = float(tuned_payload["best_threshold"])

    y_prob = best_model.predict_proba(ds.X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_tuned = (y_prob >= best_threshold).astype(int)

    test_default = metric_pack(ds.y_test, y_pred_default, y_prob)
    test_tuned = metric_pack(ds.y_test, y_pred_tuned, y_prob)

    # Calibration plot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    CalibrationDisplay.from_predictions(ds.y_test, y_prob, ax=ax1)
    ax1.set_title("Calibration Curve (Test)")
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "step5_calibration_curve_test.png", dpi=150)
    plt.close(fig1)

    # Confusion matrix plot
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(ds.y_test, y_pred_tuned, ax=ax2)
    ax2.set_title("Confusion Matrix (Test, tuned threshold)")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "step5_confusion_matrix_tuned.png", dpi=150)
    plt.close(fig2)

    cls_report = classification_report(ds.y_test, y_pred_tuned)
    (REPORTS_DIR / "step5_classification_report_tuned.txt").write_text(cls_report, encoding="utf-8")

    payload = {
        "best_model_name": tuned_payload["best_model_name"],
        "best_params": tuned_payload["best_params"],
        "best_threshold": best_threshold,
        "test_metrics_default_0_5": test_default,
        "test_metrics_tuned_threshold": test_tuned,
        "classification_report_path": str(REPORTS_DIR / "step5_classification_report_tuned.txt"),
        "plots": {
            "calibration": str(PLOTS_DIR / "step5_calibration_curve_test.png"),
            "confusion_matrix_tuned": str(PLOTS_DIR / "step5_confusion_matrix_tuned.png"),
        },
    }
    write_json(REPORTS_DIR / "step5_evaluation_report.json", payload)
    return payload


if __name__ == "__main__":
    out = run_evaluate_stage()
    print(out)
