from typing import Any

import numpy as np
import pandas as pd

from utils.proba_utils import cum_sum_intervals_from_weights, sample_cols, create_nan_vals


def forget_random_col_per_sample(remove_vals_df: pd.DataFrame,
                                 weight_map: dict[Any, int],
                                 random_generator: np.random.Generator
                                 ) -> tuple[pd.DataFrame, dict, list]:
    cum_sum_intervals, cum_sum_col_map = cum_sum_intervals_from_weights(weight_map)

    picked_cols = sample_cols(remove_vals_df.shape[0], cum_sum_intervals, cum_sum_col_map, random_generator)
    missing_vals_df, missing_col_map = create_nan_vals(remove_vals_df, picked_cols)

    missing_vals_idxs = list(missing_vals_df.index)

    return missing_vals_df, missing_col_map, missing_vals_idxs

def forget_random_col_per_sample_2(remove_vals_df: pd.DataFrame,
                                   weight_map: dict[Any, int],
                                   random_generator: np.random.Generator
                                   ) -> tuple[pd.DataFrame, dict, list, list]:
    cum_sum_intervals, cum_sum_col_map = cum_sum_intervals_from_weights(weight_map)

    picked_cols = sample_cols(remove_vals_df.shape[0], cum_sum_intervals, cum_sum_col_map, random_generator)
    missing_vals_df, missing_col_map = create_nan_vals(remove_vals_df, picked_cols)

    missing_vals_idxs = list(missing_vals_df.index)

    return missing_vals_df, missing_col_map, missing_vals_idxs, picked_cols
