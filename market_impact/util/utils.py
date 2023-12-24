import numpy as np
import pandas as pd
from scipy import stats


def _check_imbalance_validity(imbalance_column: str):
    """
    Validates if the given conditional variables are supported.
    """
    valid_imbalance_columns = ["sign_imbalance", "volume_imbalance"]
    if imbalance_column not in valid_imbalance_columns:
        raise ValueError(f"Unknown imbalance column: {imbalance_column}. Expected one of {valid_imbalance_columns}.")


def normalize_imbalances(orderbook_states: pd.DataFrame, normalization_constant: str = "daily_volume") -> pd.DataFrame:
    # TODO: add option to normalise by average volume at top of queue, make consistent with normalze_size
    data = orderbook_states.copy()

    # Rescale imbalance columns
    if "volume_imbalance" in data.columns:
        data["volume_imbalance"] = data["volume_imbalance"] / data["daily_vol"]
    if "sign_imbalance" in data.columns:
        data["sign_imbalance"] = data["sign_imbalance"] / data["daily_num"]

    return data


def normalize_conditional_impact(aggregate_impact: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale the aggregate impact 'R' in the provided DataFrame by its daily corresponding value 'R(1)'.

    The function assumes that the DataFrame contains the columns 'R' for aggregate impact and 'daily_R1'
    that defines a characteristic size or length. The normalization is performed by dividing 'R' by the
     absolute value of 'daily_R1', where we use 'abs' to preserve the sign when the average order signs
     are negative.

    Parameters:
    - aggregate_impact (pd.DataFrame): DataFrame containing the aggregate impact data.
      Expected to have columns 'R' and 'daily_R1'.

    Returns:
    - pd.DataFrame: DataFrame with the normalized 'R' values.

    Note:
    - It is assumed that 'daily_R1' represents a constant of unit dimension that define a
      character size or length (base value) for normalization.
    """

    # Creating a copy to avoid modifying the original DataFrame
    df = aggregate_impact.copy()

    # Normalizing 'R' by its daily corresponding value 'R(1)' (represented as 'daily_R1')
    if "R" in df.columns:
        # Ensure that the signs of 'R' aren't inverted
        df["R"] = df["R"] / abs(df["daily_R1"])

    return df


def bin_data_into_quantiles(df, x_col="volume_imbalance", y_col="R", q=100, duplicates="raise"):
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
    df: pd.DataFrame,
    T=None,
    columns=["volume_imbalance", "sign_imbalance", "R"],
    std_level=2,
    remove=False,
    verbose=False,
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