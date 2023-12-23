import numpy as np
import pandas as pd


# TODO: add functional form of conditional and unconditional R(1, v) and R(l) and Neural network


def scaling_function(x: np.ndarray, alpha: float, beta: float, CONST) -> np.ndarray:
    """
    Apply the scaling function to a given array.

    The scaling function :math:`\\mathcal{F}(x)` is defined as:

    .. math::
        \\mathcal{F}(x) = \\frac{x}{(1 + |x|^\\alpha)^{\\beta/\\alpha}}

    where :math:`x` is the input array, :math:`\\alpha` and :math:`\\beta` are parameters
    that shape the function and :math:`\\text{{CONST}` represents a constant factor.

    Parameters:
    - x (np.ndarray): An array of imbalances.
    - alpha (float): The alpha parameter affecting the shape of the scaling function.
    - beta (float): The beta parameter affecting the shape of the scaling function
    - CONST (float): A constant factor in the scaling function.

    Returns:
    - np.ndarray: An array of scaled imbalances.
    """
    return x / np.power(1 + np.power(np.abs(x), alpha), beta / alpha) * CONST


def scaling_form(
    orderflow_imbalance: pd.DataFrame, RN: float, QN: float, alpha: float, beta: float, CONST: float
) -> np.ndarray:
    """
    Apply the scaling form to order flow imbalances.

    The scaling form :math:`\\mathcal{S}` is defined as:

    .. math::
        \\mathcal{S}(\\text{{orderflow_imbalance}}, RN, QN, \\alpha, \\beta) = (RN \\times T) \\times
        \\mathcal{F}\\left(\\frac{{\\text{{imbalance}}}}{QN \\times T}\\right)}

    where :math:`\\mathcal{F}` is the scaling function, :math:`RN` and :math:`QN` are rescaling factors,
    :math:`\\alpha` and :math:`\\beta` are parameters of the scaling function, and :math:`\\text{{CONST}}`
    is a constant factor in the scaling form.

    Parameters:
    - orderflow_imbalance (pd.DataFrame): A DataFrame containing the columns 'volume_imbalance' and 'T'.
    - RN (float): Rescaling factor for the return scale.
    - QN (float): Rescaling factor for the volume scale.
    - alpha (float): The alpha parameter for the scaling function.
    - beta (float): The beta parameter for the scaling function.
    - CONST (float): A constant factor in the scaling form.

    Returns:
    - np.ndarray: An array of scaled order flow imbalances.
    """

    # Extract imbalance and T from the DataFrame
    T = orderflow_imbalance["T"].values

    imbalance = orderflow_imbalance["imbalance"].values

    # Apply the scaling function to the rescaled imbalance
    rescaled_imbalance = imbalance / (QN * T)
    scaled_imbalance = scaling_function(rescaled_imbalance, alpha, beta, CONST)

    return (RN * T) * scaled_imbalance * CONST


def scaling_law(
    orderflow_imbalance: pd.DataFrame, chi: float, kappa: float, alpha: float, beta: float, CONST: float
) -> np.ndarray:
    """
    Apply the scaling law to order flow imbalances.

    The scaling law :math:`R(\\text{{orderflow_imbalance}}, chi, kappa, \\alpha, \\beta)` is defined as:

    .. math::
        R(\\text{{imbalance}}, T) = T^{\\chi} \\times \\mathcal{F}\\left(\\frac{{\\text{{imbalance}}}}{T^{\\kappa}}\\right)}

    where :math:`\\mathcal{F}` is the scaling function, :math:`\\chi` and :math:`\\kappa` are scaling exponents,
    :math:`\\alpha` and :math:`\\beta` are parameters of the scaling function, and :math:`\\text{{CONST}}`
    is a constant factor in the scaling law.

    Parameters:
    - orderflow_imbalance (pd.DataFrame): A DataFrame containing the columns 'volume_imbalance' and 'T'.
    - chi (float): The chi exponent in the scaling law.
    - kappa (float): The kappa exponent in the scaling law.
    - alpha (float): The alpha parameter for the scaling function.
    - beta (float): The beta parameter for the scaling function.
    - CONST (float): A constant factor in the scaling law.

    Returns:
    - np.ndarray: An array of scaled order flow imbalances according to the scaling law.
    """

    # Extract imbalance over some T from the DataFrame
    T = orderflow_imbalance["T"].values
    imbalance = orderflow_imbalance["imbalance"].values

    # Rescale imbalance according to kappa
    rescaled_imbalance = imbalance / np.power(T, kappa)

    # Apply the scaling function
    scaled_imbalance = scaling_function(rescaled_imbalance, alpha, beta, CONST)

    return np.power(T, chi) * scaled_imbalance * CONST
