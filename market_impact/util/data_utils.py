import pylab
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import rc
from matplotlib import pyplot as plt
from typing import Any, Dict, Union, List, Optional

from market_impact.function_form import scaling_form, scaling_law

# Plot constrants
pylab.rcParams["xtick.major.pad"] = "8"
pylab.rcParams["ytick.major.pad"] = "8"
rc("text", usetex=True)
rc("mathtext", fontset="stix")
rc("axes", labelsize="large")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern} \usepackage{amssymb}"


def _validate_imbalances(imbalance_column: str) -> None:
    """Validates whether given imbalance variable is supported."""
    valid_imbalance_columns = ["sign_imbalance", "volume_imbalance"]
    if imbalance_column not in valid_imbalance_columns:
        raise ValueError(
            f"Unknown imbalance column: {imbalance_column}. Expected one of {valid_imbalance_columns}."
        )


def _determine_labels(x_col: str, plot_type: str) -> tuple:
    """
    Determines the labels for the x and y axes based on the column name and plot type.

    Args:
        x_col (str): The name of the x-axis column.
        plot_type (str): The type of plot ('form' for scaling form or 'law' for scaling law).

    Returns:
        tuple: A tuple containing the x and y axis labels.
    """
    _validate_imbalances(x_col)

    # Determine label for scaling_form and scaling_law
    if plot_type == 'scaling_form':
        if x_col == "volume_imbalance":
            return (r"$\it{\Delta V/V_{D}}$", r"$\it{R(\Delta V^\prime,T)/\mathcal{R}(1)}$")
        elif x_col == "sign_imbalance":
            return (r"$\it{\Delta \mathcal{E}^\prime/\mathcal{E}_{D}}$", r"$\it{R(\Delta \mathcal{E},T)/\mathcal{R}(1)}$")
        else:
            return (x_col, r"\it{R}")
    elif plot_type == 'scaling_law':
        if x_col == "volume_imbalance":
            return (r"$\it{\Delta V/V_{D}T^{\varkappa}}$", r"$\it{R(\Delta V^\prime,T)/\mathcal{R}(1)T^{\chi}}$")
        elif x_col == "sign_imbalance":
            return (r"$\it{\Delta \mathcal{E}^\prime/\mathcal{E}_{D}T^{\varkappa}}$", r"$\it{R(\Delta \mathcal{E}^\prime,T)/\mathcal{R}(1)T^{\chi}}$")
        else:
            return (x_col, r"\it{R}")

    # Default return if plot_type is not recognized
    return (x_col, r"\it{R}")


def normalize_aggregate_impact(aggregate_impact: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the aggregate impact 'R' in the provided DataFrame by its corresponding daily value `daily_R1`.

    Args:
        aggregate_impact (pd.DataFrame): DataFrame containing the aggregate impact data.

    Returns:
        normalized_agggregate_impact (pd.DataFrame): DataFrame with the normalized 'R' values.

    Note:
        We use 'abs()' to preserve the sign of `R` when the average order signs are negative.
        Assumes that the input DataFrame containts aggregat impact `R`, and `daily_R1` that represents a
        constant of unit dimension that define a characterstic size or length (base value) for normalization.
    """
    data = aggregate_impact.copy()

    if "R" in data.columns:
        data["R"] = data["R"] / abs(data["daily_R1"])

    return data


def normalize_imbalances(
    aggregate_features_data: pd.DataFrame,
    normalization_constant: str = "daily",
    conditional_variable: str = "volume_imbalance"
) -> pd.DataFrame:
    """
    Normalize imbalances in aggregate features DataFrame using either daily values or average queue values.

    Args:
        aggregate_features_data (pd.DataFrame): A DataFrame containing aggregate features data.
        normalization_constant (str, optional): The normalization mode - either 'daily' or 'average'. Defaults to "daily".
        conditional_variable (str, optional): The type of imbalance to normalize - either 'volume_imbalance' or
        'sign_imbalance'. Defaults to "volume_imbalance".

    Returns:
        pd.DataFrame: A DataFrame with the normalized imbalance values.

    Raises:
        ValueError: If an unknown normalization constant is provided.
    """
    # Make a copy of the input DataFrame to avoid modifying the original data
    data = aggregate_features_data.copy()

    # Check for valid imbalance column
    _validate_imbalances(imbalance_column=conditional_variable)

    # Apply normalization based on the specified constant
    if normalization_constant == "daily":
        # Normalize by daily values
        norm_column = "daily_num" if conditional_variable == "sign_imbalance" else "daily_vol"
        data[conditional_variable] = data[conditional_variable] / data[norm_column]

    elif normalization_constant == "average":
        # Normalize by average queue values
        norm_column = "average_num_at_best" if conditional_variable == "sign_imbalance" else "average_vol_at_best"
        data[conditional_variable] = data[conditional_variable] / data[norm_column]

    else:
        # Handle unknown normalization constant
        valid_constants = ['daily', 'average']
        raise ValueError(
            f"Unknown normalization constant: {normalization_constant}. Expected one of {valid_constants}."
        )

    return data


def bin_data_into_quantiles(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    q: int = 100,
    duplicates: str = "raise"
) -> pd.DataFrame:
    """
    Dynamically bins a DataFrame into quantiles based on a specified column.

    Args:
        df (pd.DataFrame): The DataFrame to be binned.
        x_col (Optional[str]): The column in 'df' on which to base the quantile bins. Defaults to None.
        y_col (Optional[str]): The column in 'df' for which the mean is calculated in each bin. Defaults to None.
        q (int): The number of quantiles to bin into. Defaults to 100.
        duplicates (str): Handling of duplicate edges (can be 'raise' or 'drop'). Defaults to "raise".

    Returns:
        pd.DataFrame: A DataFrame containing the binned 'x_col', mean of 'y_col', and, if present, the first value of 'T' in each bin.

    Note:
        The function will raise an error if 'duplicates' is set to 'raise' and duplicate bin edges are found.
    """
    # Bin 'x_col' into quantiles
    binned_x = pd.qcut(df[x_col], q=q, labels=False, retbins=True, duplicates=duplicates)
    df["x_bin"] = binned_x[0]

    # Calculate mean of 'y_col' for each bin
    y_binned = df.groupby(["x_bin"])[y_col].mean()
    y_binned.index = y_binned.index.astype(int)

    # Calculate mean of 'x_col' for each bin
    x_binned = df.groupby(["x_bin"])[x_col].mean()
    x_binned.index = x_binned.index.astype(int)

    # If 'T' column exists, include the first value of 'T' for each bin
    if "T" in df.columns:
        r_binned = df.groupby(["x_bin"])["T"].first()
        r_binned.index = r_binned.index.astype(int)
    else:
        r_binned = None

    # Concatenate the binned data into a single DataFrame
    return pd.concat([x_binned, r_binned, y_binned], axis=1).reset_index(drop=True)


def smooth_outliers(
    data: pd.DataFrame,
    columns: Optional[List[str]],
    T: Optional[int] = None,
    std_level: int = 2,
    remove: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    # TODO: default columns to None
    """
    Clip or remove values at 3 standard deviations for each series in a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to process.
        T (Optional[int], optional): An additional column indicator to append to each column in 'columns'. Defaults to None.
        columns (Optional[List[str]], optional): List of column names to process. Defaults to None.
        std_level (int, optional): The number of standard deviations to use as the clipping or removal threshold. Defaults to 2.
        remove (bool, optional): If True, rows with outliers will be removed. If False, values will be clipped. Defaults to False.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with outliers smoothed.
    """
    if T:
        all_columns = columns + [f"R{T}"]
    else:
        all_columns = columns

    all_columns = set(all_columns).intersection(data.columns)
    if len(all_columns) == 0:
        return data

    if remove:
        # Remove rows with outliers
        z = np.abs(stats.zscore(data[columns]))
        original_shape = data.shape
        data = data[(z < std_level).all(axis=1)]
        new_shape = data.shape
        if verbose:
            print(f"Removed {original_shape[0] - new_shape[0]} rows")
    else:
        # Clip values in each column
        def winsorize_queue(s: pd.Series, level) -> pd.Series:
            upper_bound = level * s.std()
            lower_bound = -level * s.std()
            if verbose:
                print(f"clipped at {upper_bound}")
            return s.clip(upper=upper_bound, lower=lower_bound)

        for name in all_columns:
            s = data[name]
            if verbose:
                print(f"Series {name}")
            data[name] = winsorize_queue(s, level=std_level)

    return data


def plot_scaling_form(
    aggregate_impact_data: pd.DataFrame,
    fit_parameters: Dict[str, float],
    q: int,
    imbalance_column: str = "volume_imbalance"
) -> None:
    """
    Plots the scaling form where the scaling function is plotted for each system size `T` before renormalization.

    Args:
        aggregate_impact_data (pd.DataFrame): A data frame containing aggregate impact data ["T", imbalance_column, "R"].
        fit_parameters (Dict[str, float]): A dictionary where keys are system sizes (T) and values are fitting params.
        q (int): The quantile into which the data is binned.
        imbalance_column (str, optional): The name of the column in `smoothed_data` to be used as the x-axis. Defaults to "volume_imbalance".
    
    Returns:
        Function does not return anything but plots a representation of the scaling form for each system size T.
    """
    data = aggregate_impact_data.copy()
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.gca()

    # Iterate over each system size and its parameters
    for T, params in fit_parameters.items():
        aggregate_impact = data[data["T"] == T][["T", imbalance_column, "R"]]
        smoothed_data = smooth_outliers(aggregate_impact, columns=[imbalance_column, "R"])
        binned_data = bin_data_into_quantiles(smoothed_data,  x_col=imbalance_column, y_col="R", q=q, duplicates="drop")

        # Prepare data for plotting
        T_values = binned_data["T"].values
        imbalance_values = binned_data[imbalance_column].values
        R_values = binned_data["R"].values
        orderflow_imbalance = pd.DataFrame({"T": T_values, "imbalance": imbalance_values})

        # Generate model predictions using the scaling_form
        model_predictions = scaling_form(orderflow_imbalance, *params)

        # Plot
        plt.scatter(x=imbalance_values, y=R_values,label=f"T = {T}", s=20)
        plt.plot(imbalance_values, model_predictions)
        ax.minorticks_off()
        xlabel, ylabel = _determine_labels(x_col=imbalance_column, plot_type="scaling_form")
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        ax.tick_params(
            bottom=True,
            top=True,
            left=True,
            right=True,
            direction="in",
            width=0.5,
            size=3,
        )
    # Legend
    legend = ax.legend(markerfirst=True)
    for indx, T in enumerate(fit_parameters.keys()):
        legend.get_texts()[indx].set_text(f"$T = {int(T)}$")
    legend.markerscale = 0.1


def plot_scaling_law(
    rescaled_parameters: Dict[Any, Any],
    q: int,
    predictions: bool = False,
    imbalance_column: str = "volume_imbalance",
) -> None:
    """
    Plots the scaling law after the scaling functions have collapsed onto a single master curve.

    This function visualizes the scaling law by plotting the rescaled parameters for different system sizes.
    It plots both the raw data points and the model predictions to demonstrate how they align to form a master curve.

    Args:
        rescaled_parameters (Dict[Any, Any]): A dictionary where keys are system sizes (T), and values are objects
        containing `params` for the scaling law and `rescaled_data` (DataFrame) to plot.
        q (int): The quantile into which the data is binned.
        predictions (bool): Whether to plot model predictions atop of collapsed data. Defaults to false.
        imbalance_column (str, optional): The name of the column in the data to be used as the x-axis. Defaults to "volume_imbalance".

    Returns:
        None: Function does not return anything but creates a plots a represntation of the scaling law.
    """
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.gca()

    # Iterate through each system size and its corresponding parameters
    for T, rescaled_param in rescaled_parameters.items():
        params = rescaled_param.params
        data = rescaled_param.data.copy()
        smoothed_data = smooth_outliers(data, columns=[imbalance_column, "R"])
        binned_data = bin_data_into_quantiles(data, x_col=imbalance_column, y_col="R", q=q, duplicates="drop")

        # Prepare data for plotting
        T_values = binned_data["T"].values
        imbalance_values = binned_data[imbalance_column].values
        R_values = binned_data["R"].values
        orderflow_imbalance = pd.DataFrame({"T": T_values, "imbalance": imbalance_values})

        # Generating model predictions using the scaling_law
        model_predictions = scaling_law(orderflow_imbalance, *params)

        # Plotting data and model predictions
        plt.scatter(x=imbalance_values, y=R_values, label=f"T = {T}", s=20)
        if predictions:
            plt.plot(imbalance_values, model_predictions)

        # Setting axis properties
        ax.minorticks_off()
        xlabel, ylabel = _determine_labels(imbalance_column, 'scaling_law')
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        ax.tick_params(
            bottom=True,
            top=True,
            left=True,
            right=True,
            direction="in",
            width=0.5,
            size=3,
        )
    # Legend
    legend = ax.legend(markerfirst=True)
    for indx, T in enumerate(rescaled_parameters.keys()):
        legend.get_texts()[indx].set_text(f"$T = {int(T)}$")
    legend.markerscale = 0.1
