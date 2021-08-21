from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def time_series_train_test_split(df: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # To be updated

    assert (frac < 1) and (frac > 0)

    df_train = df[:int(frac * len(df))].fillna(0)
    df_test = df[int(frac * len(df)):].fillna(0)

    return df_train, df_test


def data_normalization(df_train: pd.DataFrame, df_test: pd.DataFrame):

    normalizer = MinMaxScaler()

    df_train_normalized = normalizer.fit_transform(df_train)
    df_test_normalized = normalizer.transform(df_test)

    return df_train_normalized, df_test_normalized


def data_preparation(df: pd.DataFrame, history_points: int, future: int, duration: int):

    X_train = np.array([df[i: i + history_points] for i in range(len(df) - history_points)])

    y_train = np.array([df[i + history_points + future: i + history_points + future + duration, 0].sum() for i in
                        range(len(df) - history_points - future)])
    X_train = X_train[:len(y_train)]

    return X_train, y_train




