import numpy as np
import pandas as pd


def add_noise(df: pd.DataFrame, mu=0, sigma=0.1):  # mu = mean, sigma=std, los default corresponden a ruido gaussiano
    noise = np.random.normal(mu, sigma, df.shape)
    noisy_df = df + noise

    return noisy_df
