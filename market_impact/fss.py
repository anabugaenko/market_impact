import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import least_squares

from powerlaw_function import Fit

from market_impact.util.utils import bin_data_into_quantiles, _check_imbalance_validity
from market_impact.functional_form import scaling_form, scaling_law

# Fixme: have single source for least squars fitting, where we pass hyperparams as options


class ScalingFormFitResult:
    T: int
    param: List  # RN, QN
    alpha: float
    beta: float
    data: pd.DataFrame

    def __init__(self, T, param, alpha, beta, data):
        self.T = T
        self.param = param
        self.alpha = alpha
        self.beta = beta
        self.data = data


class ScalingLawFitResult:
    T: int
    params: List
    data: pd.DataFrame

    def __init__(self, T, params, data):
        self.T = T
        self.params = params
        self.data = data


def fit_scaling_form(T_values, imbalance_values, R_values, reflect_y=False):
    """
    Fits a scaling form and return found parameters
    """

    # Create orderflow imbalance DataFrame from T_values and volume_imbalance_values
    orderflow_imbalance = pd.DataFrame({"T": T_values, "imbalance": imbalance_values})

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (-1)

    # Define the residual function
    def _residuals(params, orderflow_imbalance, R_values):
        RN, QN, alpha, beta, CONST = params

        # Compute the model prediction
        predicted = scaling_form(orderflow_imbalance, RN, QN, alpha, beta, CONST)

        # Return the residuals
        return R_values - predicted

    # Initial guess for the parameters
    initial_guess = [1, 0.1, 1.2, 1.3, 1]

    # Perform least squares optimization
    result = least_squares(_residuals, initial_guess, loss="soft_l1", args=(orderflow_imbalance, R_values))

    return result.x


def fit_known_scaling_form(T_values, imbalance_values, R_values, known_alpha, known_beta, reflect_y=False):
    """
    Fits a scaling form with known parameters alpha and beta from scaling function
    """

    # Create orderflow imbalance DataFrame from T_values and volume_imbalance_values
    orderflow_imbalance = pd.DataFrame({"T": T_values, "imbalance": imbalance_values})

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (-1)

    # Define a scaling form with known parameters
    def _known_scaling_form(data: pd.DataFrame, RN: float, QN: float, CONST: float) -> float:
        """
        This version treats RN and QN as constants to be found during optimization.
        """
        return scaling_form(data, RN, QN, known_alpha, known_beta, CONST)

    def _residuals(params, data, y):
        return y - _known_scaling_form(data, *params)

    initial_guess = [0.1, 0.1, 0.1]

    result = least_squares(_residuals, initial_guess, loss="soft_l1", args=(orderflow_imbalance, R_values))

    return result.x


def fit_scaling_law(T_values, imbalance_values, R_values, reflect_y=False):
    # Construct orderflow imbalance DataFrame from T_values and volume_imbalance_values
    orderflow_imbalance = pd.DataFrame({"T": T_values, "imbalance": imbalance_values})

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (-1)

    # Define the residual function
    def _residuals(params, orderflow_imbalance, R_values):
        chi, kappa, alpha, beta, CONST = params

        # Compute the model prediction
        predicted = scaling_law(orderflow_imbalance, chi, kappa, alpha, beta, CONST)

        # Return the residuals
        return R_values - predicted

    # Initial guess for the parameters
    initial_guess = [0.8, 0.5, 0.1, 0.1, 1]

    result = least_squares(_residuals, initial_guess, loss="soft_l1", args=(orderflow_imbalance, R_values))

    return result.x


def find_scaling_exponents(fitting_method: str, xy_values: pd.DataFrame, xmin_index=10) -> Fit:
    """
    Fits the data using the specified method and returns the fitting results.
    """
    if fitting_method == "MLE":
        return Fit(xy_values, xmin_distance="BIC", xmin_index=xmin_index)
    return Fit(xy_values, nonlinear_fit_method=fitting_method, xmin_distance="BIC")


def find_shape_parameters(
    conditional_aggregate_impact: pd.DataFrame, imbalance_column: str = "volume_imbalance", reflect_y: bool = False
):
    """
    :param imbalance_column:
    :param conditional_aggregate_impact: normalized dataframe consisting {system_size_T, temperature_x, observable_A}
    which map over to observationw windows, orderflowimbalance and aggregate impact {T, imbalance, R} respective
    (where we invert f(x) along the x-axis).
    :return: shape_parameters: a list consisting of scaling form shape parameters alpha and beta.
    """

    # Data preprocessing
    data = conditional_aggregate_impact.copy()
    _check_imbalance_validity(imbalance_column)

    # Extract variables describing the  system size T, sign Δε or volume imbalance ΔV, and observable R
    R_values = data["R"].values
    T_values = data["T"].values
    imbalance_values = data[imbalance_column].values

    # Find scaling function shape parameters alpha and beta by fitting the scaling form or its reflex
    RN, QN, alpha, beta, CONST = fit_scaling_form(T_values, imbalance_values, R_values, reflect_y=reflect_y)

    # Retrieve shape parameters
    shape_params = [alpha, beta]

    return shape_params


def find_scale_factors(
    conditional_aggregate_impact_,
    alpha,
    beta,
    reflect_y=False,
    fitting_method="MLE",
    imbalance_column: str = "volume_imbalance",
    **kwargs,
):
    """
    Extract series of RT and VT from fit params for each T
    """

    # Data preprocessing
    data = conditional_aggregate_impact_.copy()
    _check_imbalance_validity(imbalance_column)

    # Fits a scaling form and returns dictionary of found scale factors R_T and V_T for each T
    scale_factors = mapout_scale_factors(data, alpha, beta, reflect_y=reflect_y, imbalance_column=imbalance_column)

    # Create a series of RT and VT from fitting the scaling form for each T.
    RT_series = []
    VT_series = []

    for lag, result in scale_factors.items():
        RT_series.append(result.param[0])
        VT_series.append(result.param[1])

    lags = list(scale_factors.keys())

    # Series of scale factors
    # FIXME: is this correct order for VT and RT series
    scaled_RT = [r * lag for r, lag in zip(RT_series, lags)]
    scaled_VT = [r * lag for r, lag in zip(VT_series, lags)]

    # Prepare data for fitting (ensure DataFrame is 2D (has exactly two columns))
    RT = pd.DataFrame({"x_values": lags, "y_values": scaled_RT})
    VT = pd.DataFrame({"x_values": lags, "y_values": scaled_VT})

    # Fit and return scaling exponents
    RT_fit_object = find_scaling_exponents(fitting_method, RT, **kwargs)
    VT_fit_object = find_scaling_exponents(fitting_method, VT, **kwargs)

    return RT, VT, RT_fit_object, VT_fit_object, scale_factors


def mapout_scale_factors(
    conditional_aggregate_impact,
    alpha,
    beta,
    reflect_y=False,
    imbalance_column: str = "volume_imbalance",
):
    """
    Helper function to map out the scale factors VT and RT as a function of T.
    """

    # Data preprocessing
    _check_imbalance_validity(imbalance_column)

    # Map-out scale factors
    scale_factors = {}
    Ts = conditional_aggregate_impact["T"].unique()
    for T in Ts:
        data = conditional_aggregate_impact[conditional_aggregate_impact["T"] == T][["T", imbalance_column, "R"]]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        # FIXME: q=len(conditional_aggregate_impact) for per data point precision
        binned_data = bin_data_into_quantiles(data, x_col=imbalance_column, duplicates="drop", q=1000)

        # Extract variables describing the system size T, sign Δε or volume imbalance ΔV, and observable R
        T_values = binned_data["T"].values
        R_values = binned_data["R"].values
        imbalance_values = binned_data[imbalance_column].values

        # Fit known scaling form
        param = fit_known_scaling_form(
            T_values=T_values,
            imbalance_values=imbalance_values,
            R_values=R_values,
            known_alpha=alpha,
            known_beta=beta,
            reflect_y=reflect_y,
        )

        scale_factors[T] = ScalingFormFitResult(
            T, param, alpha, beta, pd.DataFrame({"x": imbalance_values, "y": R_values})
        )

    return scale_factors


def transform(
    conditional_aggregate_impact: pd.DataFrame,
    master_curve_params,
    durations,
    q=100,
    reflect_y=False,
    imbalance_column: str = "volume_imbalance",
):
    """
    Used for renormalization and collapse of data at different scales. After the
    transformation, it should return similar params for different system sizes (scales).
    """

    # Data preprocessing
    df = conditional_aggregate_impact.copy()
    _check_imbalance_validity(imbalance_column)

    CHI, KAPPA, ALPHA, BETA, CONST = master_curve_params
    rescale_params = {}
    for T in durations:
        result = df[df["T"] == T][["T", imbalance_column, "R"]]

        result[imbalance_column] = result[imbalance_column] / np.power(T, KAPPA)
        result["R"] = result["R"] / np.power(T, CHI)
        new_data = bin_data_into_quantiles(result, x_col=imbalance_column, q=q, duplicates="drop")

        R_values = new_data["R"].values
        T_values = new_data["T"].values
        imbalance_values = new_data[imbalance_column].values

        # Find new (rescaled) parameters
        rescale_param = fit_scaling_law(T_values, imbalance_values, R_values, reflect_y=reflect_y)

        # Store new rescaled parameters for each T
        if rescale_param[0] is not None:
            rescale_params[T] = ScalingLawFitResult(T, rescale_param, new_data)
        else:
            print(f"Failed to fit for lag {T}")

    return rescale_params