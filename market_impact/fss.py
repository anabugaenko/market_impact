import numpy as np
import pandas as pd
from powerlaw_function import Fit
from typing import Optional, List, Dict, Tuple

from market_impact.util.utils import (
    bin_data_into_quantiles,
    _validate_imbalances,
)
from market_impact.fit import (
    ScalingFormFitResult,
    ScalingLawFitResult,
    fit_known_scaling_form,
    fit_scaling_form,
    fit_scaling_law,
)


def mapout_scale_factors(
    aggregate_impact_data: pd.DataFrame,
    alpha: float,
    beta: float,
    reflect_y: bool = False,
    imbalance_column: str = "volume_imbalance",
    initial_params: Optional[np.ndarray] = None,
) -> Dict[float, ScalingFormFitResult]:
    """
    Maps-out the scale factors RT and VT as a function of T by fitting
    the scaling form to the aggregate impact data for each T.

    Args:
        aggregate_impact_data (pd.DataFrame): DataFrame containing the aggregate impact data.
        alpha (float): Known alpha parameter from the scaling function.
        beta (float): Known beta parameter from the scaling function.
        reflect_y (bool): If True, reflects the scaling function along the x-axis. Default is False.
        imbalance_column (str): Column name in the DataFrame for the order flow imbalance data.
        initial_params (Optional[np.ndarray]): Initial guess for the scale factors. Default is None.

    Returns:
        Dict[float, ScalingFormFitResult]: A dictionary mapping each bin size N to its corresponding
        ScalingFormFitResult, including the scale factors, rescaling exponents and other relevant fitting information.

    Note:
        Fits the scaling form with known shape parameters alpha `α` and beta `β` for each unique bin size T.
    """
    scale_factors = {}
    _validate_imbalances(imbalance_column)
    Ts = aggregate_impact_data["T"].unique()
    for T in Ts:
        data = aggregate_impact_data[aggregate_impact_data["T"] == T][
            ["T", imbalance_column, "R"]
        ]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        # FIXME: q should be input param, where q=len(conditional_aggregate_impact) for per data point precision
        binned_data = bin_data_into_quantiles(
            data, x_col=imbalance_column, duplicates="drop", q=1000
        )

        # Extract observables describing the system
        R_values = data["R"].values  # physical quantity R
        T_values = data["T"].values  # system size T
        imbalance_values = data[
            imbalance_column
        ].values  # temperature (sign Δε or volume imbalance ΔV)

        # Fit scaling form with known shape parameters paramters
        param = fit_known_scaling_form(
            T_values=T_values,
            imbalance_values=imbalance_values,
            R_values=R_values,
            known_alpha=alpha,
            known_beta=beta,
            reflect_y=reflect_y,
            initial_params=initial_params,
        )

        # Store optimization results for each bin frequency T
        scale_factors[T] = ScalingFormFitResult(
            T,
            param,
            alpha,
            beta,
            pd.DataFrame({"x": imbalance_values, "y": R_values}),
        )

    return scale_factors


def find_critical_exponents(
    xy_values: pd.DataFrame, fitting_method: str, xmin_index=10
) -> Fit:
    """
    Determine scaling behaviour of the data by fitting power law and compare against alternative hypthesis.
    """
    if fitting_method == "MLE":
        return Fit(xy_values, xmin_distance="BIC", xmin_index=xmin_index)

    return Fit(xy_values, nonlinear_fit_method=fitting_method, xmin_distance="BIC")


def find_shape_parameters(
    aggregate_impact_data: pd.DataFrame,
    reflect_y: bool = False,
    initial_param: Optional[List[float]] = None,
    imbalance_column: str = "volume_imbalance",
) -> List[float]:
    """
    Find shape parameters `α` and `β` of the scaling function 𝓕(x) by fitting the scaling form to all bin frequencies `T`.

    Args:
        aggregate_impact_data (pd.DataFrame): DataFrame containing normalized data
        reflect_y (bool, optional): If True, inverts the scaling function along the x-axis. Default is False.
        initial_param (Optional[List[float]], optional): Initial guess for the fitting parameters. Default is None.
        imbalance_column (str, optional): Column name for the imbalance data. Defaults to "volume_imbalance".

    Returns:
        List[float]: A list containing the fitted shape parameters alpha and beta.

    Note:
        Asssumes aggregate impact is a DataFrame containing a representation of the system size (T), imbalance
        (either sign or volume), and aggregate impact (R) columns corresponding to ["T", "imbalance", "R"].
    """
    data = aggregate_impact_data.copy()
    _validate_imbalances(imbalance_column)

    # Extract observables describing the system
    R_values = data["R"].values  # physical quantity R
    T_values = data["T"].values  # system size T
    imbalance_values = data[
        imbalance_column
    ].values  # temperature (sign Δε or volume imbalance ΔV)

    # Find scaling function shape parameters alpha and beta by fitting the scaling form or its reflection
    RN, QN, alpha, beta = fit_scaling_form(
        T_values,
        imbalance_values,
        R_values,
        reflect_y=reflect_y,
        initial_params=initial_param,
    )
    shape_params = [alpha, beta]

    return shape_params


def find_scale_factors(
    aggregate_impact_data: pd.DataFrame,
    alpha: float,
    beta: float,
    reflect_y: bool = False,
    fitting_method: str = "MLE",
    imbalance_column: str = "volume_imbalance",
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict[float, ScalingFormFitResult]]:
    """
    Find the rescaling exponents ξ and ψ by fitting the scaling form to the shape of aggregate impact for each `T`
    while keeping values of the shape parameters alpha `α` and beta `β` the same (constant) for all `T`.

    Args:
        aggregate_impact_data (pd.DataFrame): DataFrame containing the aggregate impact data.
        alpha (float): Known alpha parameter from the scaling function.
        beta (float): Known beta parameter from the scaling function.
        reflect_y (bool): If True, reflects the scaling function along the x-axis.
        fitting_method (str): Method used for power-law fitting.
        imbalance_column (str): Column name in the DataFrame for order flow imbalance data.

    Returns:
        Tuple: Contains DataFrames for scaled RT and VT values, fit objects representing the series RT and VT,
        fits of rescaling exponents ξ and ψ, and a dictionary of scale factors for each bin size T.

    Note:
        The preceding fit of the scaling form, yielding VT and RT for each T, doesn't impose any assumptions
        on their scaling.
    """
    data = aggregate_impact_data.copy()
    _validate_imbalances(imbalance_column)

    # Map out scale factors RT and VT for each T
    scale_factors = mapout_scale_factors(
        data,
        alpha,
        beta,
        reflect_y=reflect_y,
        imbalance_column=imbalance_column,
    )

    # FIXME: is this correct order for VT and RT series
    # Retrieve rescaling exponents ξ and ψ
    RT_series = []
    VT_series = []
    for lag, result in scale_factors.items():
        RT_series.append(result.param[0])
        VT_series.append(result.param[1])

    # Perform rescaling given the corresponding bin-size T
    bin_size = list(scale_factors.keys())
    scaled_RT = [r * lag for r, lag in zip(RT_series, bin_size)]
    scaled_VT = [r * lag for r, lag in zip(VT_series, bin_size)]

    # Prepare data for fit and determine scaling behaviour of RT and VT
    RT = pd.DataFrame({"x_values": bin_size, "y_values": scaled_RT})
    VT = pd.DataFrame({"x_values": bin_size, "y_values": scaled_VT})
    RT_fit_object = find_critical_exponents(RT, fitting_method, **kwargs)
    VT_fit_object = find_critical_exponents(VT, fitting_method, **kwargs)

    return RT, VT, RT_fit_object, VT_fit_object, scale_factors


def transform(
    aggregate_impact_data: pd.DataFrame,
    master_curve_params: List[float],
    durations: List[int],
    q: int = 100,
    reflect_y: bool = False,
    imbalance_column: str = "volume_imbalance",
) -> Dict[float, ScalingLawFitResult]:
    """
    Transforms aggregate impact at different scales by rescaling the data onto a single scaling function 𝓕(x).

    Args:
        aggregate_impact_data (pd.DataFrame): DataFrame containing conditional aggregate impact data.
        master_curve_params (List[float]): Parameters of the master curve (chi, kappa, alpha, beta, CONST).
        durations (List[int]): List of durations (T) for which the data needs to be transformed.
        q (int): Number of quantiles to bin the data into for precision. Default is 100.
        reflect_y (bool): If True, inverts the scaling function along the y-axis. Default is False.
        imbalance_column (str): Column name for the order flow imbalance data.

    Returns:
        Dict[int, ScalingLawFitResult]: A dictionary mapping each duration T to its corresponding
        ScalingLawFitResult, which includes the rescaled parameters and transformed data.

    Note:
         The data should return similar shape parameters for the different binning frequencies following renormalization.
    """
    df = aggregate_impact_data.copy()
    _validate_imbalances(imbalance_column)

    chi, kappa, alpha, beta, CONST = master_curve_params
    rescale_params = {}
    for T in durations:
        result = df[df["T"] == T][["T", imbalance_column, "R"]]

        result[imbalance_column] = result[imbalance_column] / np.power(T, kappa)
        result["R"] = result["R"] / np.power(T, chi)
        new_data = bin_data_into_quantiles(
            result, x_col=imbalance_column, q=q, duplicates="drop"
        )

        # Extract observables describing the system
        R_values = new_data["R"].values  # physical quantity R
        T_values = new_data["T"].values  # system size T
        imbalance_values = new_data[
            imbalance_column
        ].values  # temperature (sign Δε or volume imbalance ΔV)

        # Retrieve new rescaled parameters
        rescale_param = fit_scaling_law(
            T_values, imbalance_values, R_values, reflect_y=reflect_y
        )

        # Store rescaled parameters and data for each bin_size T
        if rescale_param[0] is not None:
            rescale_params[T] = ScalingLawFitResult(T, rescale_param, new_data)
        else:
            print(f"Failed to fit for lag {T}")

    return rescale_params
