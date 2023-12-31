import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from typing import Any


def imputed_vals(cols : list[str], df : pd.DataFrame) -> tuple[dict, dict]:
    real_var_map_dict = dict()
    imputed_var_map_dict = dict()
    for col in cols:
        imputed_var_map_dict[col] = []
        real_var_map_dict[col] = []

    for _, sample in df.iterrows():
        imputed_col = sample["imputed"]

        real_val = sample[imputed_col + " (real)"]
        imputed_val = sample[imputed_col + " (imputed)"]

        imputed_var_map_dict[imputed_col].append(imputed_val)
        real_var_map_dict[imputed_col].append(real_val)

    return imputed_var_map_dict, real_var_map_dict

def imputed_sqr_err(cols: list[str], df: pd.DataFrame) -> dict:
    imputed_map_dict = dict()
    for col in cols:
        imputed_map_dict[col] = []

    for _, sample in df.iterrows():
        imputed_col = sample["imputed"]

        real_val = sample[imputed_col + " (real)"]
        imputed_val = sample[imputed_col + " (imputed)"]

        imputed_map_dict[imputed_col].append((real_val - imputed_val) ** 2)

    imputed_map = dict()

    for key in imputed_map_dict.keys():
        sqr_err_list = imputed_map_dict[key]
        sqr_err_len = len(sqr_err_list)
        sqr_err_sum = sum(sqr_err_list)
        mean_sqr_err = sqr_err_sum / sqr_err_len if sqr_err_len != 0 else 0
        imputed_map[key] = mean_sqr_err

    return imputed_map


def run_comparing(labeled_df: pd.DataFrame,
                  random_missing_df: pd.DateOffset,
                  missing_vals_idxs: list,
                  missing_col_per_pos: list,
                  imputer_type: str,
                  var_param: str,
                  param_type: str,
                  var_name: str,
                  var_range: list,
                  config: dict,
                  estimator_config: dict = {},
                  disable_scaling: bool = False) -> tuple[pd.DataFrame, dict]:
    mse_df_dict = {
        var_name: [],
        "col": [],
        "val": []
    }

    imputer_type_map = {
        "kNN": lambda c, estimator_config: KNNImputer(weights="uniform", **c),
        "WkNN": lambda c, estimator_config: KNNImputer(weights="distance", **c),
        "MICE": lambda c, estimator_config: IterativeImputer(estimator=LinearRegression(**estimator_config), **c),
        "MICE BR": lambda c, estimator_config: IterativeImputer(estimator=BayesianRidge(**estimator_config), **c),
        "MICE RF": lambda c, estimator_config: IterativeImputer(estimator=RandomForestRegressor(**estimator_config),
                                                                **c),
    }

    imputer_map = dict()

    if (imputer_type == "kNN" or imputer_type == "WkNN") and not disable_scaling:
        cols = set(missing_col_per_pos)
        scaler = get_scaler(labeled_df, cols)
        labeled_df = apply_scaler(labeled_df, scaler)
        random_missing_df = apply_scaler(random_missing_df, scaler)

    for curr_var_val in var_range:
        running_config = dict()
        running_config.update(config)

        running_estimator_config = dict()
        running_estimator_config.update(estimator_config)

        if param_type == "imputer":
            running_config[var_param] = curr_var_val
        elif param_type == "estimator":
            running_estimator_config[var_param] = curr_var_val

        imputer = imputer_type_map[imputer_type](running_config, running_estimator_config)
        train_df = random_missing_df.loc[random_missing_df.index.difference(missing_vals_idxs)]
        imputer = imputer.fit(train_df)

        imputer_map[curr_var_val] = imputer
        imputed_mat = imputer.transform(random_missing_df)

        imputed_df = pd.DataFrame(imputed_mat, columns=labeled_df.columns, index=labeled_df.index)
        imputed_df = imputed_df.loc[missing_vals_idxs]

        temp_labeled_df = labeled_df

        if (imputer_type == "kNN" or imputer_type == "WkNN") and not disable_scaling:
            temp_labeled_df = unscale(labeled_df, scaler)
            imputed_df = unscale(imputed_df, scaler)

        for col in labeled_df.columns:
            imputed_df["{} (real)".format(col)] = temp_labeled_df[col]
            imputed_df["{} (imputed)".format(col)] = imputed_df[col]
            imputed_df.drop([col], axis=1, inplace=True)

        imputed_df["imputed"] = missing_col_per_pos

        sqr_err_dict = imputed_sqr_err(temp_labeled_df.columns, imputed_df)

        for col in temp_labeled_df.columns:
            mse_df_dict["col"].append(col)
            mse_df_dict["val"].append(sqr_err_dict[col])
            mse_df_dict[var_name].append(curr_var_val)

    mse_df = pd.DataFrame(mse_df_dict)
    return mse_df, imputer_map

def run(
        labeled_df: pd.DataFrame,
        random_missing_df: pd.DateOffset,
        missing_vals_idxs: list,
        missing_col_per_pos: list,
        imputer_type: str,
        config: dict,
        estimator_config: dict = {}, 
        disable_scaling: bool = False) -> tuple[pd.DataFrame, Any]:
    
    imputer_type_map = {
        "kNN": lambda c, estimator_config: KNNImputer(weights="uniform", **c),
        "WkNN": lambda c, estimator_config: KNNImputer(weights="distance", **c),
        "MICE": lambda c, estimator_config: IterativeImputer(estimator=LinearRegression(**estimator_config), **c),
        "MICE BR": lambda c, estimator_config: IterativeImputer(estimator=BayesianRidge(**estimator_config), **c),
        "MICE RF": lambda c, estimator_config: IterativeImputer(estimator=RandomForestRegressor(**estimator_config),
                                                                **c),
    }

    if (imputer_type == "kNN" or imputer_type == "WkNN") and not disable_scaling:
        cols = set(missing_col_per_pos)
        scaler = get_scaler(labeled_df, cols)
        labeled_df = apply_scaler(labeled_df, scaler)
        random_missing_df = apply_scaler(random_missing_df, scaler)

    running_config = dict()
    running_config.update(config)

    running_estimator_config = dict()
    running_estimator_config.update(estimator_config)

    imputer = imputer_type_map[imputer_type](running_config, running_estimator_config)
    train_df = random_missing_df.loc[random_missing_df.index.difference(missing_vals_idxs)]
    imputer = imputer.fit(train_df)

    imputed_mat = imputer.transform(random_missing_df)

    imputed_df = pd.DataFrame(imputed_mat, columns=labeled_df.columns, index=labeled_df.index)
    imputed_df = imputed_df.loc[missing_vals_idxs]

    if (imputer_type == "kNN" or imputer_type == "WkNN") and not disable_scaling:
        labeled_df = unscale(labeled_df, scaler)
        imputed_df = unscale(imputed_df, scaler)

    for col in labeled_df.columns:
        imputed_df["{} (real)".format(col)] = labeled_df[col]
        imputed_df["{} (imputed)".format(col)] = imputed_df[col]
        imputed_df.drop([col], axis=1, inplace=True)

    imputed_df["imputed"] = missing_col_per_pos

    return imputed_df, imputer

def get_scaler(df: pd.DataFrame, cols: list[str]) -> dict:
    col_scaler = dict()

    for col in cols:
        col_max = df[col].max()
        col_min = df[col].min()
        col_range = col_max - col_min
        col_scaler[col] = (col_max, col_min, col_range)

    return col_scaler

def apply_scaler(df: pd.DataFrame, col_scaler: dict) -> pd.DataFrame:
    df = df.copy()
    for col in col_scaler.keys():
        _, col_min, col_range = col_scaler[col]
        df[col] = (df[col] - col_min) / col_range
    return df

def unscale(df: pd.DataFrame, col_scaler: dict) -> pd.DataFrame:
    df = df.copy()
    for col in col_scaler.keys():
        _, col_min, col_range = col_scaler[col]
        df[col] = df[col]*col_range + col_min
    return df