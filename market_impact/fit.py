import numpy as np
import pandas as pd
from typing import List, Optional

from market_impact.function_form import scaling_form, scaling_law
from market_impact.util.optimize import least_squares_fit


# FIXME: bring ScalingFormFitResult and ScalingLawFitResult in line with respect to constructor and return params
class ScalingFormFitResult:
    """
    Represents the result of a fitting procedure for the scaling form.

    Attributes:
        T (int): The bin size or event-time scale used in the fitting procedure.
        param (List[float]): The parameters resulting from the scaling form.
        alpha (float): The alpha parameter from the scaling function.
        beta (float): The beta parameter from the scaling function.
        data (pd.DataFrame): The data used for fitting.
    """

    T: int
    param: List  # RN, QN, alpha, beta, CONST
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
    """
     Represents the result of scaling_law fitting procedure.

    Attributes:
        T (int): The time scale or bin size used in the fitting process.
        params (List[float]): The parameters from the fitting the scaling law.
        data (pd.DataFrame): The dataset used for the fitting process.
    """

    T: int
    params: List  # chi, kappa,  alpha, beta, CONST
    data: pd.DataFrame

    def __init__(self, T, params, data):
        self.T = T
        self.params = params
        self.data = data


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
    Fits a scaling form with known parameters alpha `α` and beta `β` from the scaling function.
    """
    orderflow_imbalance = pd.DataFrame({"T": T_values, "imbalance": imbalance_values})

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (-1)

    # Define a scaling form with known parameters
    def _known_scaling_form(data: pd.DataFrame, RN: float, QN: float) -> float:
        """
        This version treats RN and QN as optimization paramters
        to be found whilst fixing alpha and beta as constants.
        """
        return scaling_form(data, RN, QN, known_alpha, known_beta)

    # Intial parameters
    if not initial_params:
        initial_params = [0.1, 0.1]

    # Perform least squares optimization
    residuals, params, fitted_values = least_squares_fit(
        x_values=orderflow_imbalance,
        y_values=R_values,
        initial_params=initial_params,
        function=_known_scaling_form,
        bounds=(-np.inf, np.inf),
    )
    result = params

    return result


def fit_scaling_form(
    T_values: List[float],
    imbalance_values: List[float],
    R_values: List[float],
    reflect_y: bool = False,
    initial_params: Optional[List[float]] = None,
) -> dict:
    """
    Fit a scaling form to the aggregate impact R(ΔV, T) data using chosen optimization method.

    Args:
        T_values (List[float]): List of binning frequencies or event-time scale values T.
        imbalance_values (List[float]): List of order flow imbalance values Δ.
        R_values (List[float]): List of aggregate impact values R.
        reflect_y (bool, optional): If True, inverts the scaling function along the x-axis. Default is False.
        initial_params (Optional[List[float]], optional): Initial guess for the scaling parameters. Default is None.

    Returns:
        dict: A dictionary containing the optimized parameters 'RN', 'QN', 'alpha', 'beta', and 'CONST'.

    Notes:
        The function uses a neural network or the method of least squares to find the optimal scaling form parameters.
    """
    orderflow_imbalance = pd.DataFrame({"T": T_values, "imbalance": imbalance_values})

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (-1)

    # Initial parameters
    if not initial_params:
        initial_params = [1, 0.1, 1, 1]

    # Perform least squares optimization
    residuals, params, fitted_values = least_squares_fit(
        x_values=orderflow_imbalance,
        y_values=R_values,
        initial_params=initial_params,
        function=scaling_form,
        bounds=(-np.inf, np.inf),
    )
    result = params
    return result


def fit_scaling_law(
    T_values, imbalance_values, R_values, reflect_y=False, initial_params=None
):
    """
     Fit a scaling law to the renormalized aggregate impact R(ΔV, T) data using chosen optimization method.

     Args:
        T_values (List[float]): List of binning frequencies or event-time scale values T.
        imbalance_values (List[float]): List of order flow imbalance values ΔV.
        R_values (List[float]): List of conditional aggregate impact values R.
        reflect_y (bool, optional): If True, reflects the scaling function along the x-axis.
        initial_params (List[float], optional): Initial guess for the scaling parameters.

    Returns:
        Dict: A dictionary containing the optimized parameters chi, kappa, alpha, beta and Consts.

    Note:
        Assumes the conditional aggregate impact data ["T", "imbalance", "R"] has been renormalized.
    """

    # Construct orderflow imbalance DataFrame from T_values and volume_imbalance_values
    orderflow_imbalance = pd.DataFrame({"T": T_values, "imbalance": imbalance_values})

    # Fit scaling form reflection where we invert the scaling function along the x-axis
    if reflect_y:
        orderflow_imbalance["imbalance"] = orderflow_imbalance["imbalance"] * (-1)

        # Model predictions
        predicted = scaling_law(orderflow_imbalance, chi, kappa, alpha, beta, CONST)

        # Return the residuals
        return R_values - predicted

    # Initial parameters
    if not initial_params:
        initial_params = [0.5, 0.5, 0.1, 0.1, 1]

    # Perform least squares optimization
    residuals, params, fitted_values = least_squares_fit(
        x_values=orderflow_imbalance,
        y_values=R_values,
        initial_params=initial_params,
        function=scaling_law,
    )
    result = params

    return result
