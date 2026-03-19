from src.data.preprocess import (
    preprocess_oil,
    preprocess_holiday,
    merge_base_features
)
from src.features.calendar_features import add_calendar_features
from src.features.lag_features import add_train_time_features
from src.features.one_hot_encode import encode_features
import pandas as pd
import numpy as np

def add_is_holiday_column(df):
    df = df.copy()
    df['is_holiday'] = (
        (df['locale'] == 'National') | 
        ((df['locale'] == 'Regional') & (df['state'] == df['locale_name'])) |
        ((df['locale'] == 'Local') & (df['city'] == df['locale_name']))
    )
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    return df

def build_train_test_features(train, test, oil, stores, holiday):
    oil = preprocess_oil(oil)
    holiday_feat = preprocess_holiday(holiday)

    train, test = merge_base_features(train, test, oil, stores, holiday_feat)

    train = add_is_holiday_column(train)
    test = add_is_holiday_column(test)

    train = add_calendar_features(train)
    test = add_calendar_features(test)

    # train = add_train_time_features(train)

    train_encoded, test_encoded = encode_features(train, test)
    return train_encoded, test_encoded

def build_inference_features(input_df, oil, stores, holiday, feature_columns):
    oil = preprocess_oil(oil)
    holiday_feat = preprocess_holiday(holiday)

    dummy_train = pd.DataFrame(columns=input_df.columns)
    input_df["date"] = pd.to_datetime(input_df["date"])
    _, inference_df = merge_base_features(dummy_train, input_df.copy(), oil, stores, holiday_feat)

    inference_df = add_is_holiday_column(inference_df)
    inference_df = add_calendar_features(inference_df)

    categorical_cols = ["family", "city", "state", 'type_x', 'type_y', 'locale', 'locale_name']
    inference_df = pd.get_dummies(inference_df, columns=categorical_cols)

    # LightGBM dislikes special characters in feature names
    inference_df.columns = inference_df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)

    inference_df = inference_df.reindex(columns=feature_columns, fill_value=0)
    
    return inference_df