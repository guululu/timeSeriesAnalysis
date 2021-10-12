import os

import pandas as pd
import numpy as np

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


def split_dataset(data, week: int, n_gap: int, n_out: int, scale: int = 100000, scaler=None):
    # split into standard weeks

    data = data.copy()

    train, test = data[:-week * 7], data[-(week + n_gap + n_out) * 7:]
    y_train, y_test = train[:, 0] / scale, test[:, 0] / scale

    train = train / scale
    test = test / scale

    return train, test, y_train, y_test


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
