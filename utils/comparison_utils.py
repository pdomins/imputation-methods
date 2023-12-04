import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compare_imputations(df_real, df_imputed):
    comparison_results = []
    for column in df_real.columns:
        mse = mean_squared_error(df_real[column], df_imputed[column])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df_real[column], df_imputed[column])
        comparison_results.append({
            'Column': column,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        })
    return pd.DataFrame(comparison_results)
