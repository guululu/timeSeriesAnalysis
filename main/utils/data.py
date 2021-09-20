import os
from typing import List

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from main.io.path_definition import get_project_dir


def load_data() -> pd.DataFrame:

    filename = os.path.join(get_project_dir(), "data", "raw", "原始資料.xlsx")
    xls = pd.ExcelFile(filename, engine='openpyxl')
    df = pd.read_excel(xls, index_col=0)

    df = df[['代號', '數量']]

    df.reset_index(inplace=True)
    df_sum = df.groupby(['代號', '交易日期'], as_index=False)['數量'].sum()

    # Identify unique 代號
    unique_product_code = np.unique(df_sum['代號'])

    product_df_list = []

    # Replace all the 數量 with the 代號
    for code in unique_product_code:
        temp = df_sum[df_sum['代號'] == code].set_index("交易日期")
        temp.drop(labels=['代號'], inplace=True, axis=1)
        temp.rename(columns={'數量': code}, inplace=True)
        product_df_list.append(temp)

    df_parsed = pd.concat(product_df_list, axis=1)

    return df_parsed


def covid_country(df, iso_code):
    covid19 = df[df['iso_code'] == iso_code]

    date = [pd.Timestamp(covid19['date'].iloc[i]) for i in range(len(covid19))]

    columns = ['total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed',
               'icu_patients',
               'hosp_patients']

    data = covid19[columns].values

    columns = [f'{iso_code}_{c}' for c in columns]

    df_covid = pd.DataFrame(data=data, columns=columns, index=date)

    df_covid['week_day'] = [idx.weekday() for idx in df_covid.index]
    df_covid[f'{iso_code}_new_cases_7_sum'] = df_covid[f'{iso_code}_new_cases'].rolling(7, min_periods=1).sum()
    df_covid[f'{iso_code}_new_deaths_7_sum'] = df_covid[f'{iso_code}_new_deaths'].rolling(7, min_periods=1).sum()
    df_covid[f'{iso_code}_new_cases_3_sum'] = df_covid[f'{iso_code}_new_cases'].rolling(3, min_periods=1).sum()
    df_covid[f'{iso_code}_new_deaths_3_sum'] = df_covid[f'{iso_code}_new_deaths'].rolling(3, min_periods=1).sum()

    for idx, row in df_covid.iterrows():
        if row['week_day'] == 0:
            idx_start = idx
            break

    for idx, row in df_covid.iloc[::-1].iterrows():
        if row['week_day'] == 6:
            idx_end = idx
            break

    df_covid = df_covid.loc[idx_start: idx_end]

    df_covid.drop(labels=['week_day'], axis=1, inplace=True)

    return df_covid.fillna(0)


def split_dataset(data, numerical_columns_index: List, week: int, n_gap: int, n_out: int, scale: int = 100000):
    # split into standard weeks

    train, test = data[:-week * 7], data[-(week + n_gap + n_out) * 7:]
    y_train, y_test = train[:, 0] / scale, test[:, 0] / scale

    train_norm = train[:, numerical_columns_index] / scale
    test_norm = test[:, numerical_columns_index] / scale

    train_category = np.delete(train, numerical_columns_index, axis=1)
    test_category = np.delete(test, numerical_columns_index, axis=1)

    # merge categorical data with normalized numerical data
    train_norm = np.concatenate((train_norm, train_category), axis=1)
    test_norm = np.concatenate((test_norm, test_category), axis=1)

    # restructure into windows of weekly data
    train_norm = np.array(np.split(train_norm, len(train_norm) / 7))
    test_norm = np.array(np.split(test_norm, len(test_norm) / 7))
    return train_norm, test_norm, y_train, y_test


def load_stock_data(code: int):
    # yahoo finance 下載股票

    data = yf.Ticker(f'{code}.TW')

    df = data.history(start='2017-01-01', end='2021-07-31')

    return df


def raw_data_preparation():
    df_parsed = load_data()

    df = df_parsed.copy()
    df = df.resample('1D').sum()
    df.fillna(0, inplace=True)

    df['week_day'] = [idx.weekday() for idx in df.index]

    for idx, row in df.iterrows():
        if row['week_day'] == 0:
            idx_start = idx
            break  # 跳出 for loop

    for idx, row in df.iloc[::-1].iterrows():  # 從後面數回來
        if row['week_day'] == 6:
            idx_end = idx
            break

    return df.loc[idx_start: idx_end]
