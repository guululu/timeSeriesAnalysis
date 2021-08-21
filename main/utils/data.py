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