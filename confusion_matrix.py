import pandas as pd
import numpy as np
from typing import Any
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns


def __init_confusion_matrix_rows__(out_labels: np.ndarray) -> dict[Any, np.ndarray]:
    out_labels_size = out_labels.shape[0]
    return dict(map(lambda x: (x, np.zeros(out_labels_size)), out_labels))


def get_confusion_matrix_row(conf_mat: pd.DataFrame, real: Any) -> pd.DataFrame:
    return conf_mat.loc[real].to_frame().T


def get_confusion_matrix_value(conf_mat: pd.DataFrame, real: Any, predicted: Any) -> float:
    return conf_mat[predicted][real]


def calculate_confusion_matrix(possible_out_labels: np.ndarray, predicted: dict[Any, Any],
                               expected: dict[Any, Any]) -> pd.DataFrame:
    data = __init_confusion_matrix_rows__(possible_out_labels)
    conf_mat_df = pd.DataFrame(columns=possible_out_labels, index=possible_out_labels, data=data)
    for expected_idx, expected_label in expected.items():
        predicted_label = predicted[expected_idx]
        conf_mat_df[predicted_label][expected_label] += 1
    return conf_mat_df


def calculate_relative_confusion_matrix(possible_out_labels: np.ndarray, predicted: dict[Any, Any],
                                        expected: dict[Any, Any]) -> pd.DataFrame:
    conf_mat_df = calculate_confusion_matrix(possible_out_labels, predicted, expected)
    return calculate_relative_confusion_matrix_from_confusion_matrix(conf_mat_df)


def calculate_relative_confusion_matrix_from_confusion_matrix(confusion_matrix: pd.DataFrame) -> pd.DataFrame:
    real_count = confusion_matrix.sum(axis='columns')
    real_count = real_count.replace(0, 1)
    rel_conf_mat_df = confusion_matrix.div(real_count, axis='index')
    return rel_conf_mat_df


def calculate_true_positives_from_confusion_matrix(confusion_matrix: pd.DataFrame, positive_label: Any) -> float:
    return confusion_matrix.loc[positive_label][positive_label]


def calculate_false_positives_from_confusion_matrix(confusion_matrix: pd.DataFrame, positive_label: Any) -> float:
    return confusion_matrix.drop([positive_label], axis="index")[[positive_label]].values.sum()


def calculate_false_negatives_from_confusion_matrix(confusion_matrix: pd.DataFrame, positive_label: Any) -> float:
    return confusion_matrix.drop([positive_label], axis="columns").loc[positive_label].values.sum()


def calculate_true_negatives_from_confusion_matrix(confusion_matrix: pd.DataFrame, positive_label: Any) -> float:
    return confusion_matrix.drop([positive_label], axis="index").drop([positive_label], axis="columns").values.sum()


def calculate_true_positive_rate_from_confusion_matrix(confusion_matrix: pd.DataFrame, positive_label: Any) -> float:
    TP = calculate_true_positives_from_confusion_matrix(confusion_matrix, positive_label)
    FN = calculate_false_negatives_from_confusion_matrix(confusion_matrix, positive_label)
    return TP / (TP + FN)


def calculate_false_positive_rate_from_confusion_matrix(confusion_matrix: pd.DataFrame, positive_label: Any) -> float:
    FP = calculate_false_positives_from_confusion_matrix(confusion_matrix, positive_label)
    TN = calculate_true_negatives_from_confusion_matrix(confusion_matrix, positive_label)
    return FP / (FP + TN)


def __init_per_label_confusion_matrix__() -> pd.DataFrame:
    return pd.DataFrame(data={"P": np.zeros(2),
                              "N": np.zeros(2)
                              }, index=["P", "N"])


def calculate_per_label_confusion_matrix(possible_out_labels: np.ndarray, predicted: dict[Any, Any],
                                         expected: dict[Any, Any]) -> pd.DataFrame:
    conf_mat = calculate_confusion_matrix(possible_out_labels, predicted, expected)
    return calculate_per_label_confusion_matrix_from_confusion_matrix(conf_mat)


def calculate_per_label_confusion_matrix_from_confusion_matrix(confusion_matrix: pd.DataFrame) -> pd.DataFrame:
    per_label_conf_mats = dict()
    for possible_out_label in confusion_matrix.columns:
        curr_label_conf_mat = __init_per_label_confusion_matrix__()
        curr_label_conf_mat.loc["P"]["P"] = calculate_true_positives_from_confusion_matrix(confusion_matrix,
                                                                                           possible_out_label)
        curr_label_conf_mat.loc["N"]["P"] = calculate_false_positives_from_confusion_matrix(confusion_matrix,
                                                                                            possible_out_label)
        curr_label_conf_mat.loc["P"]["N"] = calculate_false_negatives_from_confusion_matrix(confusion_matrix,
                                                                                            possible_out_label)
        curr_label_conf_mat.loc["N"]["N"] = calculate_true_negatives_from_confusion_matrix(confusion_matrix,
                                                                                           possible_out_label)
        per_label_conf_mats[possible_out_label] = curr_label_conf_mat
    return per_label_conf_mats


def count_total_samples_from_confusion_matrix(confusion_matrix: pd.DataFrame) -> int:
    return confusion_matrix.values.sum()


def calculate_confusion_matrix_correct(confusion_matrix: pd.DataFrame) -> int:
    return np.diag(confusion_matrix).sum()


def __confusion_matrix_error_by_complement__(total_samples: int, confusion_matrix: pd.DataFrame):
    correct_samples = calculate_confusion_matrix_correct(confusion_matrix)
    error = (total_samples - correct_samples)
    return error


def calculate_confusion_matrix_error(confusion_matrix: pd.DataFrame) -> int:
    total_samples = count_total_samples_from_confusion_matrix(confusion_matrix)
    return __confusion_matrix_error_by_complement__(total_samples, confusion_matrix)


def calculate_confusion_matrix_error_rate(confusion_matrix: pd.DataFrame) -> float:
    total_samples = count_total_samples_from_confusion_matrix(confusion_matrix)
    error = __confusion_matrix_error_by_complement__(total_samples, confusion_matrix)
    return error / total_samples


def metrics(per_label_conf_mats):
    metrics_per_label = {}

    for label, label_conf_mat in per_label_conf_mats.items():
        TP = calculate_true_positives_from_confusion_matrix(label_conf_mat, "P")
        FP = calculate_false_positives_from_confusion_matrix(label_conf_mat, "P")
        FN = calculate_false_negatives_from_confusion_matrix(label_conf_mat, "P")
        TN = calculate_true_negatives_from_confusion_matrix(label_conf_mat, "P")

        precision = TP / (TP + FP) if TP != 0 else 0
        recall = TP / (TP + FN) if TP != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision * recall) != 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        metrics_per_label[label] = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "Accuracy": accuracy
        }
    return metrics_per_label


def calculate_roc_confusion_matrices(prediction_probabilities: dict[Any, dict[str, Any]], expected: dict[Any, Any],
                                     thresholds: np.ndarray, positive_label: Any = "P", negative_label: Any = "N") -> \
        list[dict[str, Any]]:
    conf_mats = []
    for threshold in thresholds:
        positive_prediction_probas = dict()
        for prediction_proba_idx in prediction_probabilities.keys():
            proba_positive_scaled = prediction_probabilities[prediction_proba_idx][positive_label]
            proba_negative_scaled = prediction_probabilities[prediction_proba_idx][negative_label]
            true_positive_proba = proba_positive_scaled / (proba_positive_scaled + proba_negative_scaled)
            positive_prediction_probas[prediction_proba_idx] = true_positive_proba
        predicted = dict()
        for pos_predict_proba_idx in positive_prediction_probas.keys():
            predicted[pos_predict_proba_idx] = positive_label if positive_prediction_probas[
                                                                     pos_predict_proba_idx] >= threshold \
                else negative_label
        conf_mats.append({
            "threshold": threshold,
            "confusion_matrix": calculate_confusion_matrix(np.array([positive_label, negative_label]), predicted,
                                                           expected)
        })
    return conf_mats


def calculate_roc_positive_rates(roc_confusion_matrices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive_rates = []
    for i in range(len(roc_confusion_matrices)):
        conf_mat = roc_confusion_matrices[i]["confusion_matrix"]
        TPR = calculate_true_positive_rate_from_confusion_matrix(conf_mat, "P")
        FPR = calculate_false_positive_rate_from_confusion_matrix(conf_mat, "P")
        conf_mat_metrics = metrics({
            "metrics": conf_mat
        })
        positive_rates.append({
            "threshold": roc_confusion_matrices[i]["threshold"],
            "TPR": TPR,
            "FPR": FPR,
            "metrics": conf_mat_metrics["metrics"]
        })
    return positive_rates


# TODO: Revisar si esta permitido usar AUC de Sklearn
def calculate_auc_from_positive_rates(positive_rates: list[dict[str, Any]]) -> float:
    FPRs, TPRs = np.array(list(map(lambda pr: pr["FPR"], positive_rates))), np.array(
        list(map(lambda pr: pr["TPR"], positive_rates)))
    return auc(FPRs, TPRs)


def calculate_roc_positive_rates_uniquie(roc_positive_rates: list[dict[str, Any]]) -> dict[dict[str, Any]]:
    positive_rates = dict()
    for i in range(len(roc_positive_rates)):
        TPR = roc_positive_rates[i]["TPR"]
        FPR = roc_positive_rates[i]["FPR"]
        curr_coordinate = (FPR, TPR)
        if curr_coordinate not in positive_rates:
            positive_rates[curr_coordinate] = {
                "threshold": {
                    "min": roc_positive_rates[i]["threshold"],
                    "max": roc_positive_rates[i]["threshold"]
                },
                "TPR": TPR,
                "FPR": FPR,
                "metrics": roc_positive_rates[i]["metrics"]
            }
        else:
            if roc_positive_rates[i]["threshold"] > positive_rates[curr_coordinate]["threshold"]["max"]:
                positive_rates[curr_coordinate]["threshold"]["max"] = roc_positive_rates[i]["threshold"]
            elif roc_positive_rates[i]["threshold"] < positive_rates[curr_coordinate]["threshold"]["min"]:
                positive_rates[curr_coordinate]["threshold"]["min"] = roc_positive_rates[i]["threshold"]
    return positive_rates

def plot_confusion_matrix(confusion_matrix: pd.DataFrame, title: str, save_file: str = None, format: str = ""):
    plt.figure()
    min_val = 0
    max_val = confusion_matrix.to_numpy().max()
    ax = sns.heatmap(confusion_matrix, vmin=min_val, vmax=max_val, annot=True, fmt=format, linewidth=1,
                     cmap="Blues")
    ax.set(xlabel="Predicción", ylabel="Real", title=title)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=1200)
    else:
        plt.show()
