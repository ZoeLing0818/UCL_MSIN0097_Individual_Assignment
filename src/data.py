from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from .settings import DATA_PATH, SEED

TARGET_COL = "Attrition_Flag"
ID_COL = "CLIENTNUM"


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def prepare_dataset() -> DatasetSplit:
    raw_data = load_raw_data()

    leakage_cols = [
        c
        for c in raw_data.columns
        if c.startswith("Naive_Bayes_Classifier")
        or ("Attrition_Flag" in c and c != TARGET_COL)
    ]

    cols_to_drop = [TARGET_COL, ID_COL] + leakage_cols
    X = raw_data.drop(columns=[c for c in cols_to_drop if c in raw_data.columns]).copy()
    y = raw_data[TARGET_COL].map({"Existing Customer": 0, "Attrited Customer": 1})

    if y.isna().any():
        raise ValueError("Target contains unknown labels.")

    # Turn implicit missing token into NaN for imputation.
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].replace("Unknown", np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    # Basic schema checks.
    missing_num = [c for c in NUMERICAL_FEATURES if c not in X_train.columns]
    missing_cat = [c for c in CATEGORICAL_FEATURES if c not in X_train.columns]
    if missing_num or missing_cat:
        raise ValueError(f"Missing features: num={missing_num}, cat={missing_cat}")

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
