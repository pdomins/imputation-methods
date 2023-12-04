import numpy as np
import pandas as pd


def add_noise(df: pd.DataFrame, mu=0, sigma=0.1):  # mu = mean, sigma = std, los default corresponden a ruido gaussiano
    noise = np.random.normal(mu, sigma, df.shape)
    noisy_df = df + noise

    return noisy_df


def add_noise_to_column(df: pd.DataFrame, column_name: str):
    std_deviation = df[column_name].std()
    mean = df[column_name].mean()

    return add_noise(df[column_name], mu=mean, sigma=std_deviation)
