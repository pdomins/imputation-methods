import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_mapped_indices(missing_vals_idxs : list, new_index : list):
    missing_vals_idxs_dict = dict(map(lambda i,j : (i,j), missing_vals_idxs, new_index))
    return list(map(missing_vals_idxs_dict.get, missing_vals_idxs))

def compare_imputations(df_real : pd.DataFrame, df_imputed : pd.DataFrame, index_column_dict : dict):
    df_real = df_real.reset_index(drop=True)
    df_imputed = df_imputed.reset_index(drop=True)
    comparison_results = []
    column_index_dict = {}
    missing_vals_idxs_dict = dict(map(lambda i,j : (i,j), index_column_dict.keys(), df_imputed.index))
    for i, v in index_column_dict.items():
        column_index_dict[v] = [missing_vals_idxs_dict[i]] if v not in column_index_dict.keys() else column_index_dict[v] + [missing_vals_idxs_dict[i]]
    for column in column_index_dict.keys():
        indexes = column_index_dict[column]
        mse = mean_squared_error(df_real.iloc[indexes][column], df_imputed.iloc[indexes][column])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df_real.iloc[indexes][column], df_imputed.iloc[indexes][column])
        comparison_results.append({
            'Column': column,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        })
    return pd.DataFrame(comparison_results)
