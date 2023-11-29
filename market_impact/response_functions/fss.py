import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import least_squares

from powerlaw_function import Fit

from market_impact.util.utils import bin_data_into_quantiles
from market_impact.response_functions.functional_form import scaling_form, scaling_law


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


def fit_known_scaling_form(orderflow_imbalance, R_values, known_alpha, known_beta):
    """
    Fits scaling form with known parameters from scaling function
    """

    def _known_scaling_form(data: pd.DataFrame, RN: float, QN: float, CONST: float) -> float:
        """
        This version treats RN and QN as constants to be found during optimisation.
        """
        return scaling_form(data, RN, QN, known_alpha, known_beta, CONST)

    def _residuals(params, data, y):
        return y - _known_scaling_form(data, *params)

    initial_guess = [0.1, 0.1, 0.1]

    result = least_squares(
        _residuals,
        initial_guess,
        args=(orderflow_imbalance, R_values),
        loss="soft_l1")

    return result.x


def fit_scaling_form(T_values, vol_imbalance_values, R_values):
    # Create DataFrame from T_values and vol_imbalance_values
    orderflow_imbalance = pd.DataFrame({'T': T_values, 'vol_imbalance': vol_imbalance_values})

    # Define the residuals function
    def _residuals(params, orderflow_imbalance, R_values):
        RN, QN, alpha, beta, CONST = params

        # Calculate the model prediction
        predicted = scaling_form(orderflow_imbalance, RN, QN, alpha, beta, CONST)

        # Return the residuals
        return R_values - predicted

    # Initial guess for the parameters
    initial_guess = [1, 0.1, 1.2, 1.3, 1]

    # Perform least squares optimization
    result = least_squares(
        _residuals,
        initial_guess,
        args=(orderflow_imbalance, R_values),
        bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]),
        loss='soft_l1'
    )

    # result = least_squares(_residuals, initial_guess, args=(T_values, vol_imbalance_values, R_values), loss="soft_l1")

    return result.x

def fit_scaling_law(T_values, vol_imbalance_values, R_values):

    def residuals(params, T_values, vol_imbalance_values, R_values):

        chi, kappa, alpha, beta, CONST = params

        orderflow_imbalance = pd.DataFrame({'T': T_values, 'vol_imbalance': vol_imbalance_values})
        predicted = scaling_law(orderflow_imbalance, chi, kappa, alpha, beta, CONST)
        return R_values - predicted

    initial_guess = [0.8, 0.5, 0.1, 0.1, 1]

    result = least_squares(
        residuals,
        initial_guess,
        args=(T_values, vol_imbalance_values, R_values),
        #bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]),
        loss='soft_l1',
    )

    return result.x


def map_scale_factors(conditional_aggregate_imapct, alpha, beta):
    scale_factors = {}
    Ts = conditional_aggregate_imapct["T"].unique()
    for T in Ts:
        data = conditional_aggregate_imapct[conditional_aggregate_imapct["T"] == T][["vol_imbalance", "R", "T"]]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        binned_data = bin_data_into_quantiles(data, duplicates="drop", q=1000) # should be size of data for per data point precision

        T_values = binned_data['T'].values
        vol_imbalance_values = binned_data['vol_imbalance'].values
        R_values = binned_data['R'].values

        # Orderflow imbalance
        orderflow_imbalance = pd.DataFrame({'T': T_values, 'vol_imbalance': vol_imbalance_values})

        param = fit_known_scaling_form(orderflow_imbalance, R_values, known_alpha=alpha, known_beta=beta)
        scale_factors[T] = ScalingFormFitResult(T, param, alpha, beta, pd.DataFrame({"x": vol_imbalance_values, "y": R_values}))

    return scale_factors


def find_shape_parameters(normalized_data: pd.DataFrame):
    """
    :param normalized_data: normalized dataframe consisting {system_size_T, temperature_x, observable_A}
    which map over to observationw windows, orderflowimbalance and aggregate impact {T, imbalance, R} respective.
    :return: shape_parameters: a list consisting of scaling form shape parameters alpha and beta.
    """
    data = normalized_data.copy()

    # Extract variables describing the system
    T_values = data['T'].values
    vol_imbalance_values = data['vol_imbalance'].values
    R_values = data['R'].values

    # Find scaling form shape parameters
    RN, QN, alpha, beta, CONST = fit_scaling_form(T_values, vol_imbalance_values, R_values)
    shape_params = [alpha, beta]

    return shape_params


def find_scaling_exponents(fitting_method: str, xy_values: pd.DataFrame, xmin_index=10) -> Fit:
    """Fits the data using the specified method and returns the fitting results."""
    if fitting_method == "MLE":
        return Fit(xy_values, xmin_distance="BIC", xmin_index=xmin_index)
    return Fit(xy_values, nonlinear_fit_method=fitting_method, xmin_distance="BIC")


def find_scale_factors(conditional_aggregate_impact, alpha, beta, fitting_method="MLE", **kwargs):
    """
    Helper function to extract series of RN and QN
    from fit param for each N
    """

    # fit rescaled form at each N, returns dictionary of fitting results
    scale_factors = map_scale_factors(conditional_aggregate_impact, alpha, beta)

    RN_series = []
    QN_series = []
    # FIXME: is this correct order for QN and RN series
    for lag, result in scale_factors.items():
        RN_series.append(result.param[0])
        QN_series.append(result.param[1])


    lags = list(scale_factors.keys())

    # Series of scale factors
    scaled_RN = [r * lag for r, lag in zip(RN_series, lags)]
    scaled_QN = [r * lag for r, lag in zip(QN_series, lags)]

    # Prepare data for fitting (ensure DataFrame has exactly is 2D (has two columns))
    RN = pd.DataFrame({"x_values": lags, "y_values": scaled_RN})
    QN = pd.DataFrame({"x_values": lags, "y_values": scaled_QN})

    # Fit and return scaling exponents
    RN_fit_object = find_scaling_exponents(fitting_method, RN, **kwargs)
    QN_fit_object = find_scaling_exponents(fitting_method, QN, **kwargs)

    return RN, QN, RN_fit_object, QN_fit_object, scale_factors


def transform(conditional_aggregate_impact: pd.DataFrame, master_curve_params, durations, q=100):
    """
    Used for renormalisatio and collapse of data at different scales.
    After the transformation, should return similar params for different scales.
    """
    df = conditional_aggregate_impact.copy()

    CHI, KAPPA, ALPHA, BETA, CONST = master_curve_params
    rescale_params = {}
    for T in durations:
        result = df[df["T"] == T][["T", "vol_imbalance", "R"]]

        result["vol_imbalance"] = result["vol_imbalance"] / np.power(T, KAPPA)
        result["R"] = result["R"] / np.power(T, CHI)
        binned_data = bin_data_into_quantiles(result, q=q, duplicates="drop")

        T_values = binned_data['T'].values
        vol_imbalance_values = binned_data['vol_imbalance'].values
        R_values = binned_data['R'].values

        rescale_param = fit_scaling_law(T_values, vol_imbalance_values, R_values)

        if rescale_param[0] is not None:
            rescale_params[T] = ScalingLawFitResult(T, rescale_param, binned_data)
        else:
            print(f"Failed to fit for lag {T}")

    return rescale_params

