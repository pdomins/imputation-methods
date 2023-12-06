import math

import matplotlib.pyplot as plt
import pandas as pd


def sturges(size: int) -> int:
    return 1 + math.ceil(math.log2(size))


def plot_col_histograms(real_data: pd.DataFrame, imputed_data: pd.DataFrame, method: str):
    for col in real_data.columns:
        fig, ax = plt.subplots(1, 2)
        fig.tight_layout(pad=5.0)
        ax[0].hist(x=imputed_data[col], bins=sturges(len(imputed_data[col])))
        ax[1].hist(x=real_data[col], bins=sturges(len(real_data[col])))
        imputed_ylim = ax[0].get_ylim()
        real_ylim = ax[1].get_ylim()
        true_ylim_max = max([imputed_ylim[1], real_ylim[1]])
        real_xlim = ax[1].get_xlim()
        min_val = real_xlim[0]
        max_val = real_xlim[1]
        ax[0].grid(True, linestyle='--', linewidth=0.5)
        ax[1].grid(True, linestyle='--', linewidth=0.5)
        ax[0].set_xlim((min_val, max_val))
        ax[1].set_xlim((min_val, max_val))
        ax[0].set_ylim((0, true_ylim_max))
        ax[1].set_ylim((0, true_ylim_max))
        ax[0].set_title("Imputed")
        ax[1].set_title("Real")
        plt.suptitle(f"{col} ({method})")
        plt.show()


def plot_col_boxplots(real_data: pd.DataFrame, imputed_data: pd.DataFrame, method: str):
    for col in imputed_data.columns:
        _, ax = plt.subplots(1, 2)
        ax[0].boxplot(x=imputed_data[col])
        ax[1].boxplot(x=real_data[col])
        imputed_ylim = ax[0].get_ylim()
        real_ylim = ax[1].get_ylim()
        ylim_min = min([imputed_ylim[0], real_ylim[0]])
        ylim_max = max([imputed_ylim[1], real_ylim[1]])
        ax[0].set_ylim((ylim_min, ylim_max))
        ax[1].set_ylim((ylim_min, ylim_max))
        ax[0].set_title("Imputed")
        ax[1].set_title("Real")
        plt.suptitle(f"{col} ({method})")
        plt.show()
