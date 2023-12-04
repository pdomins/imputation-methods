import numpy as np
import pandas as pd
import math


def __shuffled_df_copy__(df: pd.DataFrame, random_state: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df = df.sample(frac=1) if random_state is None \
        else df.sample(frac=1, random_state=random_state)
    return df


def __create_k_empty_dfs_from_template__(template_df: pd.DataFrame, k: int) -> list[pd.DataFrame]:
    dfs = []
    for _ in range(k):
        dfs.append(pd.DataFrame(columns=template_df.columns))
    return dfs


def __calculate_fold_times__(df: pd.DataFrame, k: int) -> int:
    df_size = df.shape[0]
    fold_times = math.ceil(df_size / k)
    return fold_times


def __generate_k_folds__(df: pd.DataFrame, k: int, random_state: np.random.Generator = None) -> list[pd.DataFrame]:
    if (k < 2): raise ValueError("k must be >= 2. The value of k was: {}".format(k))
    df = __shuffled_df_copy__(df, random_state)
    folds = __create_k_empty_dfs_from_template__(df, k)
    fold_times = __calculate_fold_times__(df, k)
    for i in range(fold_times):
        curr_fold = df.iloc[i * k:(i + 1) * k]
        for fold_idx in range(k):
            folds[fold_idx] = pd.concat([folds[fold_idx], curr_fold.iloc[fold_idx].to_frame().T]) if curr_fold.shape[
                                                                                                         0] > fold_idx \
                else folds[fold_idx]
    return folds


def __generate_n_splits__(df: pd.DataFrame, k: int, folds: list[pd.DataFrame], n: int = None,
                          random_state: np.random.Generator = None) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if n is None:
        n = k
    if n < 1 or n > k:
        raise ValueError("n must be >= 1 and <= {}(=k). The value of n was: {}".format(k, n))
    if random_state is None:
        random_state = np.random.default_rng()

    splits = []
    folds_idxs = np.array(range(k))
    chosen_folds_idxs = random_state.choice(folds_idxs, size=n, replace=False)

    for chosen_fold_idx in chosen_folds_idxs:
        test_df = folds[chosen_fold_idx].copy()
        train_df = pd.DataFrame(columns=df.columns)
        for fold_idx in folds_idxs:
            if fold_idx != chosen_fold_idx: train_df = pd.concat([train_df, folds[fold_idx]])
        splits.append((train_df, test_df))

    return splits


def k_fold_split(df: pd.DataFrame, k: int, random_state: np.random.Generator = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    [(train_df, test_df)] = k_fold_n_splits(df, k, n=1, random_state=random_state)
    return train_df, test_df


def k_fold_n_splits(df: pd.DataFrame, k: int, n: int = None, random_state: np.random.Generator = None
                    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    folds = __generate_k_folds__(df, k, random_state)
    return __generate_n_splits__(df, k, folds, n, random_state)
