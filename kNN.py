import random
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def euclidean_distance(point1, point2) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


def calculate_distances(train_df: pd.DataFrame,
                        test_row: pd.Series,
                        attr_to_predict: Any,
                        k: int) -> list[tuple[Any, float]]:
    distances = []
    for idx, train_row in train_df.iterrows():
        dist = euclidean_distance(train_row.drop(attr_to_predict), test_row.drop(attr_to_predict))
        distances.append((idx, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:k]


def find_max_prediction(predictions: dict[Any, tuple[float, Any]]) -> tuple[Any, list[Any]]:
    max_key = max(predictions, key=lambda k: predictions[k])
    max_count = predictions[max_key]
    max_counts = [v for _, v in predictions.items() if v == max_count]
    if max_counts.count(max_count) > 1:
        return None, max_counts
    else:
        return max_key, max_counts


def get_predictions_cat(distances: list[tuple[Any, float]],
                        train_df: pd.DataFrame,
                        attr_to_predict: Any,
                        is_weighted: bool) -> tuple[Any, list[Any]]:
    predictions = defaultdict(float)
    for idx, dist in distances:
        prediction_val = train_df.loc[idx, attr_to_predict]
        if is_weighted:
            predictions[prediction_val] += 1 / (dist ** 2) if dist > 0 else float("inf")
        else:
            predictions[prediction_val] += 1
    return find_max_prediction(predictions)


def get_predictions_qual(distances: list[tuple[Any, float]],
                         train_df: pd.DataFrame,
                         attr_to_predict: Any,
                         is_weighted: bool) -> tuple[Any, list[Any]]:
    if is_weighted:
        weights = [(idx, 1 / (dist ** 2)) for idx, dist in distances]
        prediction_vals = [train_df.loc[idx, attr_to_predict] * weight for idx, weight in weights]
        dist_sum = sum(map(lambda w: w[1], weights))
        pred_avg = sum(prediction_vals) / dist_sum
        return pred_avg, [pred_avg]

    prediction_vals = [train_df.loc[idx, attr_to_predict] for idx, _ in distances]
    pred_avg = sum(prediction_vals) / len(prediction_vals)
    return pred_avg, [pred_avg]


def get_predictions(distances: list[tuple[Any, float]],
                    train_df: pd.DataFrame,
                    attr_to_predict: Any,
                    attr_type: str,
                    is_weighted: bool) -> tuple[Any, list[Any]]:
    if attr_type == "qualitative":
        return get_predictions_qual(distances, train_df, attr_to_predict, is_weighted)

    if attr_type == "categorical":
        return get_predictions_cat(distances, train_df, attr_to_predict, is_weighted)

    raise ValueError(f"attr type for {attr_to_predict} must have either categorical or qualitative attr type")


def get_prediction_for_test_row(train_df: pd.DataFrame,
                                test_row: pd.Series,
                                attr_to_predict: Any,
                                attr_type: str,
                                k: int, is_weighted: bool) -> Any:
    distances = calculate_distances(train_df, test_row, attr_to_predict, k)
    prediction, max_vals = get_predictions(distances, train_df, attr_to_predict, attr_type, is_weighted)
    new_k = k + 1
    while prediction is None and new_k <= train_df.shape[0]:
        distances = calculate_distances(train_df, test_row, attr_to_predict, new_k)
        prediction, max_vals = get_predictions(distances, train_df, attr_to_predict, attr_type, is_weighted)
        new_k += 1
    if new_k > train_df.shape[0] and prediction is None:
        prediction = random.choice(max_vals)
    return prediction


def kNN(train_df: pd.DataFrame, test_df: pd.DataFrame,
        missing_vals: dict, attr_types: dict[Any, str],
        k: int, is_weighted: bool = False) -> pd.DataFrame:
    if k > len(train_df):
        raise ValueError(f"k should be smaller than the total amount of points {len(train_df)}")

    test_df_copy = test_df.copy()
    predictions = [get_prediction_for_test_row(train_df, test_row, missing_vals[test_row_idx],
                                               attr_types[missing_vals[test_row_idx]], k, is_weighted)
                   for test_row_idx, test_row in test_df_copy.iterrows()]

    test_df_copy['predictions'] = predictions
    return test_df_copy
