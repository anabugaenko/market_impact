from typing import List

import pandas as pd
from powerlaw_function import Fit
from scipy.optimize import least_squares
import numpy as np

# TODO: single source of Fitting methods, Least-squares and Neural network.
# TODO: move residual calculation to least_squares_fit
# FIXME: bounds seem to work for scaling law but not the scaling form
# FIXME: bring ScalingFormFitResult and ScalingLawFitResult in line with respect to constructor and return params


class ScalingFormFitResult:
    """
    Represents the result of a fitting procedure for the scaling form.

    Attributes:
        T (int): The bin size or time scale used in the fitting procedure.
        param (List[float]): The parameters resulting from the fit, RN, QN, alpha, beta, and CONST.
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
        params (List[float]): The parameters from the fit, including chi, kappa, alpha, beta, CONST.
        data (pd.DataFrame): The dataset used for the fitting process.
    """

    T: int
    params: List  # chi, kappa,  alpha, beta, CONST
    data: pd.DataFrame

    def __init__(self, T, params, data):
        self.T = T
        self.params = params
        self.data = data


def least_squares_fit(residuals_func, initial_params, xs, ys):
    num_param = len(initial_params)
    result = least_squares(
        residuals_func,
        initial_params,
        loss="soft_l1",
        args=(xs, ys),
    )  # bounds=([0]*num_param, [np.inf]*num_param),

    return result.x


def powerlaw_fit(
    fitting_method: str, xy_values: pd.DataFrame, xmin_index=10
) -> Fit:
    """
    Determine scaling behaviour of the data  by fitting power law and comparing alternative heavy-tailed distributions
    """
    if fitting_method == "MLE":
        return Fit(xy_values, xmin_distance="BIC", xmin_index=xmin_index)

    return Fit(
        xy_values, nonlinear_fit_method=fitting_method, xmin_distance="BIC"
    )
