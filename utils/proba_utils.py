from typing import Any

import numpy as np
import pandas as pd


def proba_map_from_weights(weight_map: dict[Any, int]
                           ) -> dict[Any, float]:
    weight_sum = 0
    for weight in weight_map.values():
        weight_sum += weight

    proba_map = dict()
    for col, weight in weight_map.items():
        proba_map[col] = weight / weight_sum

    return proba_map


def cum_sum_intervals_from_proba_map(proba_map: dict[Any, float]
                                     ) -> tuple[list[pd.Interval], dict[int, Any]]:
    proba_map_last_item = len(proba_map) - 1

    cum_sum_col_map = dict()
    cum_sum_intervals = list()

    i = 0
    cum_sum = 0
    for col, proba in proba_map.items():
        next_cum_sum = cum_sum + proba

        closed = "left"
        if i == proba_map_last_item:
            closed = "both"
            next_cum_sum = 1

        interval = pd.Interval(left=cum_sum, right=next_cum_sum, closed=closed)

        cum_sum_intervals.append(interval)
        cum_sum_col_map[i] = col

        cum_sum = next_cum_sum
        i += 1

    return cum_sum_intervals, cum_sum_col_map


def cum_sum_intervals_from_weights(weight_map: dict[Any, int]
                                   ) -> tuple[list[pd.Interval], dict[int, Any]]:
    proba_map = proba_map_from_weights(weight_map)
    return cum_sum_intervals_from_proba_map(proba_map)


def get_interval_col(picked_val: float, cum_sum_intervals: list[pd.Interval],
                     cum_sum_col_map: dict[int, Any]) -> Any:
    interval_count = len(cum_sum_intervals)

    for i in range(interval_count):
        interval = cum_sum_intervals[i]

        if picked_val in interval:
            picked_col = cum_sum_col_map[i]
            return picked_col


def sample_cols(picked_size: int, cum_sum_intervals: list[pd.Interval],
                cum_sum_col_map: dict[int, Any],
                random_generator: np.random.Generator) -> list[Any]:
    picked = random_generator.random(picked_size)
    picked_cols = [get_interval_col(sample, cum_sum_intervals, cum_sum_col_map) for sample in picked]
    return picked_cols


def create_nan_vals(remove_vals_df: pd.DataFrame, picked_cols: list
                    ) -> tuple[pd.DataFrame, dict]:
    missing_vals_df = remove_vals_df.copy()
    missing_col_map = dict()
    i = 0
    for idx in missing_vals_df.index:
        missing_col = picked_cols[i]
        missing_col_map[idx] = missing_col

        missing_vals_df.loc[idx][missing_col] = np.NaN

        i += 1

    return missing_vals_df, missing_col_map
