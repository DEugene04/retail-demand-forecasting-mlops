import pandas as pd

def add_train_time_features(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    train = train.sort_values(["store_nbr", "family", "date"])

    grp = train.groupby(["store_nbr", "family"])["sales"]

    train["lag_1"] = grp.shift(1)
    train["lag_7"] = grp.shift(7)

    # rolling averages from past values only
    train["rolling_mean_7"] = grp.shift(1).rolling(7).mean()
    train["rolling_mean_30"] = grp.shift(1).rolling(30).mean()

    # remove rows that cannot form the lag / rolling features
    train = train.dropna(subset=["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_30"])

    return train