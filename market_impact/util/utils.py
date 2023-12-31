import numpy as np
import pandas as pd
from scipy import stats

# Helper functions

def _validate_imbalances(imbalance_column: str):
    """Validates if the given conditional imbalance variables is supported."""
    valid_imbalance_columns = ["sign_imbalance", "volume_imbalance"]
    if imbalance_column not in valid_imbalance_columns:
        raise ValueError(f"Unknown imbalance column: {imbalance_column}. Expected one of {valid_imbalance_columns}.")

def normalize_imbalances(
    orderbook_states: pd.DataFrame,
    normalization_constant: str = "daily",
    conditional_variable: str = "volume_imbalance",
) -> pd.DataFrame:
    """
    Rescale imblance each day by corresponding daily values signed volume V_D, daily number ε_D, or avg volume at best V_best.
    """

    # Data preprocessing
    data = orderbook_states.copy()
    _validate_imbalances(imbalance_column=conditional_variable)

    # Rescale using correspinding values
    if normalization_constant == "daily":
        if "volume_imbalance" in data.columns:
            data["volume_imbalance"] = data["volume_imbalance"] / data["daily_vol"]
        if "sign_imbalance" in data.columns:
            data["sign_imbalance"] = data["sign_imbalance"] / data["daily_num"]
    elif normalization_constant == "volume_at_best":
        if "volume_imbalance" in data.columns:
            data["volume_imbalance"] = data["volume_imbalance"] / data["average_vol_at_best"]
    else:
        raise ValueError(
            f"Unknown normalization constant: {normalization_constant}. Expected one of {['daily','volume_at_best']}."
        )

    return data


def normalize_aggregate_impact(aggregate_impact: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale the aggregate impact 'R' in the provided DataFrame by its corresponding daily value 'R(1)'.

    Args:
        aggregate_impact (pd.DataFrame): DataFrame containing the aggregate impact data.

    Returns:
        normalized_agggregate_impact (pd.DataFrame): DataFrame with the normalized 'R' values.

    Note:
        We use 'abs' to preserve the sign when the average order signs are negative.
        Assumed that the input DataFrame containts aggregat impact `R`, and `daily_R1` that represents a
        constant of unit dimension that define a characterstic size or length (base value) for normalization.
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
    Dynmaically bins a series of data using quantile binning
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
