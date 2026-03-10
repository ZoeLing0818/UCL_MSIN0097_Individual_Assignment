from __future__ import annotations

import json
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from .data import prepare_dataset
from .preprocess import build_pipeline
from .settings import MODELS_DIR, REPORTS_DIR, SEED
from .train import run_train_stage
from .utils import write_json


def _load_or_create_shortlist() -> dict:
    p = REPORTS_DIR / "step4_shortlist.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return run_train_stage()


def run_tune_stage() -> dict:
    ds = prepare_dataset()
    shortlist_payload = _load_or_create_shortlist()
    best_model_name = shortlist_payload["best_model_name"]

    base_models_for_tuning = {
        "random_forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(early_stopping=True, random_state=SEED),
    }

    parameter_spaces = {
        "random_forest": {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.02, 0.04, 0.08, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
    }

    if best_model_name not in parameter_spaces:
        raise ValueError(f"No parameter space configured for model: {best_model_name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    space = parameter_spaces[best_model_name]
    n_candidates = int(np.prod([len(v) for v in space.values()]))

    search = RandomizedSearchCV(
        estimator=build_pipeline(base_models_for_tuning[best_model_name]),
        param_distributions=space,
        n_iter=min(20, n_candidates),
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        cv=cv,
        random_state=SEED,
    )
    search.fit(ds.X_train, ds.y_train)
    best_model = search.best_estimator_

    tuned_template = build_pipeline(base_models_for_tuning[best_model_name])
    tuned_template.set_params(**search.best_params_)
    y_train_oof_prob = cross_val_predict(
        tuned_template,
        ds.X_train,
        ds.y_train,
        n_jobs=-1,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    thresholds = np.linspace(0.10, 0.90, 81)
    f1s = [f1_score(ds.y_train, (y_train_oof_prob >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])

    model_payload = {
        "best_model_name": best_model_name,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "best_threshold": best_threshold,
        "model": best_model,
    }

    with (MODELS_DIR / "step5_best_model.pkl").open("wb") as f:
        pickle.dump(model_payload, f)

    report_payload = {
        "best_model_name": best_model_name,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "best_threshold": best_threshold,
    }
    write_json(REPORTS_DIR / "step5_tuning_report.json", report_payload)
    return report_payload


if __name__ == "__main__":
    out = run_tune_stage()
    print(out)
