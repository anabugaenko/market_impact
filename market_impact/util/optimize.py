import warnings
from typing import List, Callable, Any, Union, Tuple

import numpy as np
from scipy.optimize import least_squares

# TODO: add gadient based optimization method for neural network .
# FIXME: least square bounds seem to be required for scaling law but for the scaling form, it depends.


def least_squares_fit(
    x_values: List[float],
    y_values: List[float],
    initial_params: List[float],
    function: Callable,
    bounds: Any = None,
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fits a nonlinear function (curve) to the data using the method of least squares.

    Parameters:
    x_values (List[float]): The independent variable values.
    y_values (List[float]): The dependent variable values.
    initial_params (List(float)): Initial guess on independent variable.
    function (Callable): The function to fit.
    bounds (Any): 2-tuple of array_like or `Bounds`, optional

    Returns:
    np.ndarray: The residuals.
    np.ndarray: The optimized parameters.
    np.ndarray: The fitted values.

    """
    num_params = function.__code__.co_argcount - 1  # Exclude the first argument

    def _residuals(
        params: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
    ) -> np.ndarray:
        model_predictions = function(x_values, *params)
        return y_values - model_predictions

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # Set bounds
            if bounds:
                bounds = bounds
            else:
                bounds = ([0] * num_params, [np.inf] * num_params)

            result = least_squares(
                _residuals,
                initial_params,
                args=(x_values, y_values),
                loss="soft_l1",
                bounds=bounds,
            )
            params = result.x

        fitted_values = function(x_values, *params)
        residuals = y_values - fitted_values
        return residuals, params, fitted_values
    except RuntimeError as e:
        print(f"Failed to fit curve for function {function.__name__}. Error: {e}")
        return None
