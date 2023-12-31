import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

from market_impact.functional_form import scaling_form, scaling_law
from market_impact.util.utils import (
    bin_data_into_quantiles,
    _validate_imbalances,
)
from market_impact.util.fit import (
    least_squares_fit,
    powerlaw_fit,
    ScalingFormFitResult,
    ScalingLawFitResult,
)


def fit_known_scaling_form(
    T_values: List[float],
    imbalance_values: List[float],
    R_values: List[float],
    known_alpha: float,
    known_beta: float,
    reflect_y: bool = False,
    initial_params: Optional[List[float]] = None,
) -> dict:
    """
    Fits a scaling form with known parameters alpha `α` and beta `β` for the scaling function.
    """

    # Create orderflow imbalance DataFrame from T_values and volume_imbalance_values
    orderflow_imbalance = pd.DataFrame(
        {"T": T_values, "imbalance": imbalance_values}
    )

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (
            -1
        )

    # Define a scaling form with known parameters
    def _known_scaling_form(
        data: pd.DataFrame, RN: float, QN: float, CONST: float
    ) -> float:
        """
        This version treats RN and QN as optimization paramters to be found whilst fixing alpha and beta as constants.
        """
        return scaling_form(data, RN, QN, known_alpha, known_beta, CONST)

    def _residuals(params, data, y):
        return y - _known_scaling_form(data, *params)

    if not initial_params:
        initial_guess = [0.1, 0.1, 0.1]
    else:
        initial_guess = initial_params

    # Perform least squares optimization
    result = least_squares_fit(
        residuals_func=_residuals,
        initial_params=initial_guess,
        xs=orderflow_imbalance,
        ys=R_values,
    )
    return result


def fit_scaling_form(
    T_values: List[float],
    imbalance_values: List[float],
    R_values: List[float],
    reflect_y: bool = False,
    initial_params: Optional[List[float]] = None,
) -> dict:
    """
    Fit a scaling form to the conditional aggregate impact R(ΔV, T) data using chosen optimization method.

    Args:
        T_values (List[float]): List of time scale values T.
        imbalance_values (List[float]): List of order flow imbalance values Δ.
        R_values (List[float]): List of aggregate impact values R.
        reflect_y (bool, optional): If True, inverts the scaling function along the x-axis. Default is False.
        initial_params (Optional[List[float]], optional): Initial guess for the scaling parameters. Default is None.

    Returns:
        dict: A dictionary containing the optimized parameters 'RN', 'QN', 'alpha', 'beta', and 'CONST'.

    Notes:
        The function uses a neural network or the method of least squares to find the optimal scaling form parameters.
    """

    # Create orderflow imbalance DataFrame from T_values and volume_imbalance_values
    orderflow_imbalance = pd.DataFrame(
        {"T": T_values, "imbalance": imbalance_values}
    )

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (
            -1
        )

    # Define the residual function for least_squares fit
    def _residuals(params, orderflow_imbalance, R_values):
        RN, QN, alpha, beta, CONST = params
        predicted = scaling_form(
            orderflow_imbalance, RN, QN, alpha, beta, CONST
        )
        return R_values - predicted

    # Initial guess for the parameters
    if not initial_params:
        initial_guess = [1, 0.1, 1.2, 1.3, 1]
    else:
        initial_guess = initial_params

    # Perform least squares optimization
    result = least_squares_fit(
        residuals_func=_residuals,
        initial_params=initial_guess,
        xs=orderflow_imbalance,
        ys=R_values,
    )
    return result


def fit_scaling_law(
    T_values, imbalance_values, R_values, reflect_y=False, initial_params=None
):
    """
     Fit a scaling law to the renormalized conditional aggregate impact R(ΔV, T) data using chosen optimization method.

     Args:
        T_values (List[float]): List of time scale values T.
        imbalance_values (List[float]): List of order flow imbalance values ΔV.
        R_values (List[float]): List of conditional aggregate impact values R.
        reflect_y (bool, optional): If True, reflects the scaling function along the x-axis.
        initial_params (List[float], optional): Initial guess for the scaling parameters.

    Returns:
        Dict: A dictionary containing the optimized parameters and additional information from the fitting process.

    Note:
        Assumes the conditional aggregate impact data ["T", "imbalance", "R"] has been renormalized.
    """

    # Construct orderflow imbalance DataFrame from T_values and volume_imbalance_values
    orderflow_imbalance = pd.DataFrame(
        {"T": T_values, "imbalance": imbalance_values}
    )

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (
            -1
        )

    # Define the residual function
    def _residuals(params, orderflow_imbalance, R_values):
        chi, kappa, alpha, beta, CONST = params

        # Compute the model prediction
        predicted = scaling_law(
            orderflow_imbalance, chi, kappa, alpha, beta, CONST
        )

        # Return the residuals
        return R_values - predicted

    # Initial guess for the parameters
    if not initial_params:
        initial_guess = [0.8, 0.5, 0.1, 0.1, 1]
    else:
        initial_guess = initial_params

    result = least_squares_fit(
        residuals_func=_residuals,
        initial_params=initial_guess,
        xs=orderflow_imbalance,
        ys=R_values,
    )

    return result


def mapout_scale_factors(
    aggregate_impact_data: pd.DataFrame,
    alpha: float,
    beta: float,
    reflect_y: bool = False,
    imbalance_column: str = "volume_imbalance",
    initial_params: Optional[np.ndarray] = None,
) -> Dict[float, ScalingFormFitResult]:
    """
    Maps out the scale factors and RT and VT as a function of T by fitting the scaling form to the aggregate impact data for each T.

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

    # Map-out scale factors
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

        # Extract variables describing observables (system size T, imbalance, and physical quantity R)
        T_values = binned_data["T"].values
        R_values = binned_data["R"].values
        imbalance_values = binned_data[imbalance_column].values

        # Fit scaling form with known shape parameters paramters α and β
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


def find_shape_parameters(
    conditional_aggregate_impact: pd.DataFrame,
    reflect_y: bool = False,
    initial_param: Optional[List[float]] = None,
    imbalance_column: str = "volume_imbalance",
) -> List[float]:
    """
    Find shape parameters `α` and `β` of the scaling function `𝓕(x)` by fitting the saling form to all bin frequencies `T`.

    Args:
        conditional_aggregate_impact (pd.DataFrame): DataFrame containing normalized data
        reflect_y (bool, optional): If True, inverts the scaling function along the x-axis. Default is False.
        initial_param (Optional[List[float]], optional): Initial guess for the fitting parameters. Default is None.
        imbalance_column (str, optional): Column name for the imbalance data. Defaults to "volume_imbalance".

    Returns:
        List[float]: A list containing the fitted shape parameters alpha and beta.

    Note:
        Asssumes conditional_aggregate_impact is a DataFrame containing system size (T), imbalance
        (either sign or volume), and aggregate impact (R) columns corresponding to ["T", "imblance", "R"].
    """
    # Data preprocessing
    data = conditional_aggregate_impact.copy()
    _validate_imbalances(imbalance_column)

    # Extract variables describing the  system size T, sign Δε or volume imbalance ΔV, and observable R
    R_values = data["R"].values
    T_values = data["T"].values
    imbalance_values = data[imbalance_column].values

    # Find scaling function shape parameters alpha and beta by fitting the scaling form or its reflect
    RN, QN, alpha, beta, CONST = fit_scaling_form(
        T_values,
        imbalance_values,
        R_values,
        reflect_y=reflect_y,
        initial_params=initial_param,
    )

    # Retrieve shape parameters
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
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict, Dict, Dict[float, ScalingFormFitResult]
]:
    """
    Find the rescaling exponents ξ and ψ by fitting the scaling form to the shape of aggregate impact for each T
    while keeping values of the shape parameters `α` and beta `β` the same (constant) for all T.

    Args:
        aggregate_impact_data (pd.DataFrame): DataFrame containing the aggregate impact data.
        alpha (float): Known alpha parameter from the scaling function.
        beta (float): Known beta parameter from the scaling function.
        reflect_y (bool): If True, reflects the scaling function along the x-axis.
        fitting_method (str): Method used for power-law fitting.
        imbalance_column (str): Column name in the DataFrame for order flow imbalance data.

    Returns:
        Tuple: Contains DataFrames for scaled RN and QN values, fit objects for RN and QN,
        fits of rescaling exponents ξ and ψ, and a dictionary of scale factors for each bin size T.
    """
    # Data preprocessing
    data = aggregate_impact_data.copy()
    _validate_imbalances(imbalance_column)

    # Fits a _known_scaling_form and returns dictionary of found scale factors RT and VT for each T
    scale_factors = mapout_scale_factors(
        data,
        alpha,
        beta,
        reflect_y=reflect_y,
        imbalance_column=imbalance_column,
    )

    # Create a series of RT and VT from fitting the scaling form for each T.
    # FIXME: is this correct order for VT and RT series
    RT_series = []
    VT_series = []

    for lag, result in scale_factors.items():
        RT_series.append(result.param[0])
        VT_series.append(result.param[1])

    # Perform rescaling given the correpsonding bin-size T
    bin_size = list(scale_factors.keys())
    scaled_RT = [r * lag for r, lag in zip(RT_series, bin_size)]
    scaled_VT = [r * lag for r, lag in zip(VT_series, bin_size)]

    # Prepare data for fit and determine behaviour of rescaling expontent ξ and ψ
    RT = pd.DataFrame({"x_values": bin_size, "y_values": scaled_RT})
    VT = pd.DataFrame({"x_values": bin_size, "y_values": scaled_VT})
    RT_fit_object = powerlaw_fit(fitting_method, RT, **kwargs)
    VT_fit_object = powerlaw_fit(fitting_method, VT, **kwargs)

    return RT, VT, RT_fit_object, VT_fit_object, scale_factors


def transform(
    aggregate_impact_data: pd.DataFrame,
    master_curve_params,
    durations,
    q=100,
    reflect_y=False,
    imbalance_column: str = "volume_imbalance",
):
    """
    Transforms aggregate impact data at different scales onto a single scaling function (the master curve).
    The data should return similar paramters for different binning frequencies folling the renormalization.

    Args:
        conditional_aggregate_impact (pd.DataFrame): DataFrame containing conditional aggregate impact data.
        master_curve_params (List[float]): Parameters of the master curve (CHI, KAPPA, ALPHA, BETA, CONST).
        durations (List[int]): List of durations (T) for which the data needs to be transformed.
        q (int): Number of quantiles to bin the data into for precision. Default is 100.
        reflect_y (bool): If True, inverts the scaling function along the y-axis. Default is False.
        imbalance_column (str): Column name for the order flow imbalance data.

    Returns:
        Dict[int, ScalingLawFitResult]: A dictionary mapping each duration T to its corresponding
        ScalingLawFitResult, which includes the rescaled parameters and transformed data.
    """

    # Data preprocessing
    df = aggregate_impact_data.copy()
    _validate_imbalances(imbalance_column)

    CHI, KAPPA, ALPHA, BETA, CONST = master_curve_params
    rescale_params = {}
    for T in durations:
        result = df[df["T"] == T][["T", imbalance_column, "R"]]

        result[imbalance_column] = result[imbalance_column] / np.power(T, KAPPA)
        result["R"] = result["R"] / np.power(T, CHI)
        new_data = bin_data_into_quantiles(
            result, x_col=imbalance_column, q=q, duplicates="drop"
        )

        # Prepare data for fitting
        R_values = new_data["R"].values
        T_values = new_data["T"].values
        imbalance_values = new_data[imbalance_column].values

        # New (rescaled) parameters
        rescale_param = fit_scaling_law(
            T_values, imbalance_values, R_values, reflect_y=reflect_y
        )

        # Store new rescaled parameters and data for each T
        if rescale_param[0] is not None:
            rescale_params[T] = ScalingLawFitResult(T, rescale_param, new_data)
        else:
            print(f"Failed to fit for lag {T}")

    return rescale_params
