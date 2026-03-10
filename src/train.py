from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .data import prepare_dataset
from .preprocess import build_pipeline
from .settings import SEED, REPORTS_DIR, TABLES_DIR
from .utils import write_json


def run_train_stage() -> dict:
    ds = prepare_dataset()

    models = {
        "baseline_dummy": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            max_iter=300,
            early_stopping=True,
            random_state=SEED,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": make_scorer(f1_score),
        "recall": make_scorer(recall_score),
    }

    rows = []
    for name, model in models.items():
        pipeline = build_pipeline(model)
        out = cross_validate(
            pipeline,
            ds.X_train,
            ds.y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(out["test_roc_auc"].mean()),
                "cv_roc_auc_std": float(out["test_roc_auc"].std()),
                "cv_pr_auc_mean": float(out["test_pr_auc"].mean()),
                "cv_pr_auc_std": float(out["test_pr_auc"].std()),
                "cv_f1_mean": float(out["test_f1"].mean()),
                "cv_f1_std": float(out["test_f1"].std()),
                "cv_recall_mean": float(out["test_recall"].mean()),
                "cv_recall_std": float(out["test_recall"].std()),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        ["cv_roc_auc_mean", "cv_pr_auc_mean"], ascending=False
    ).reset_index(drop=True)

    candidate_df = comparison_df[comparison_df["model"] != "baseline_dummy"].copy()
    shortlist_models = candidate_df.head(2)["model"].tolist()
    best_model_name = candidate_df.iloc[0]["model"]

    comparison_df.to_csv(TABLES_DIR / "step4_model_comparison.csv", index=False)

    payload = {
        "shortlist_models": shortlist_models,
        "best_model_name": best_model_name,
        "table_path": str(TABLES_DIR / "step4_model_comparison.csv"),
    }
    write_json(REPORTS_DIR / "step4_shortlist.json", payload)
    return payload


if __name__ == "__main__":
    out = run_train_stage()
    print(out)
