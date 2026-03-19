import pandas as pd
import numpy as np

def encode_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    categorical_cols = ["family", "city", "state", 'type_x', 'type_y', 'locale', 'locale_name']

    # add placeholder sales to test so concatenation is easy
    test["sales"] = np.nan
    # test["lag_1"] = np.nan
    # test["lag_7"] = np.nan
    # test["rolling_mean_7"] = np.nan
    # test["rolling_mean_30"] = np.nan

    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=categorical_cols)

    # LightGBM dislikes special characters in feature names
    combined.columns = combined.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)

    train_encoded = combined[combined["sales"].notna()].copy()
    test_encoded = combined[combined["sales"].isna()].copy()

    return train_encoded, test_encoded