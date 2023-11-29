import numpy as np
import pandas as pd
from scipy import stats


def normalize_aggregate_impact(aggregate_impact: pd.DataFrame):
    # Rescale R/R(1)

    # Data
    df = aggregate_impact.copy()
    if "R" in df.columns:
        df["R"] = df["R"] / df["daily_R1"]

    return df


def normalize_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Normalise by average volume at top of queue
    if "vol_imbalance" in df.columns:
        df["vol_imbalance"] = df["vol_imbalance"] / df["daily_vol"]
    if "sign_imbalance" in df.columns:
        df["sign_imbalance"] = df["sign_imbalance"] / df["daily_num"]
    return df


def bin_data_into_quantiles(df, x_col="vol_imbalance", y_col="R", q=100, duplicates="raise"):
    """
    Returns binned series.
    """
    binned_x = pd.qcut(df[x_col], q=q, labels=False, retbins=True, duplicates=duplicates)
    binned_x = binned_x[0]
    df["x_bin"] = binned_x

    y_binned = df.groupby(["x_bin"])[y_col].mean()
    y_binned.index = y_binned.index.astype(int)

    x_binned = df.groupby(["x_bin"])[x_col].mean()
    x_binned.index = x_binned.index.astype(int)

    if "T" in df.columns:
        r_binned = df.groupby(["x_bin"])["T"].first()
        r_binned.index = r_binned.index.astype(int)
    else:
        r_binned = None

    return pd.concat([x_binned, r_binned, y_binned], axis=1).reset_index(drop=True)


def smooth_outliers(
    df: pd.DataFrame, T=None, columns=["vol_imbalance", "sign_imbalance", "R"], std_level=2, remove=False, verbose=False
):
    # TODO: default columns to None
    """
    Clip or remove values at 3 standard deviations for each series.
    """
    if T:
        columns_all = columns + [f"R{T}"]
    else:
        columns_all = columns

    columns_all = set(columns_all).intersection(df.columns)
    if len(columns_all) == 0:
        return df

    if remove:
        z = np.abs(stats.zscore(df[columns]))
        original_shape = df.shape
        df = df[(z < std_level).all(axis=1)]
        new_shape = df.shape
        if verbose:
            print(f"Removed {original_shape[0] - new_shape[0]} rows")
    else:

        def winsorize_queue(s: pd.Series, level) -> pd.Series:
            upper_bound = level * s.std()
            lower_bound = -level * s.std()
            if verbose:
                print(f"clipped at {upper_bound}")
            return s.clip(upper=upper_bound, lower=lower_bound)

        for name in columns_all:
            s = df[name]
            if verbose:
                print(f"Series {name}")
            df[name] = winsorize_queue(s, level=std_level)

    return df


