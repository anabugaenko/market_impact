import numpy as np
import pandas as pd


# TODO: add functional form of unconditional impact R(l) and Neural network
def scaling_function(x: np.ndarray, alpha: float, beta: float, CONST) -> np.ndarray:
    """
    Apply the scaling function :math:`\mathcal{F}(x)` to a given series, defined as a sigmoidal:

    .. math::
        \mathcal{F}(x) = \frac{x}{(1 + |x|^\alpha)^{\beta/\alpha}}

    where :math:`x` is the input array, :math:`\alpha` and :math:`\beta` are parameters
    that shape the function and :math:`\text{{CONST}` represents a constant factor.

    Args:
        x (np.ndarray): An array of imbalances.

    Parameters:
        alpha (float): The alpha parameter affecting the shape of the scaling function.
        beta (float): The beta parameter affecting the shape of the scaling function
        CONST (float): A constant factor in the scaling function.

    Returns:
        np.ndarray: An array of scaled imbalances.
    """
    return x / np.power(1 + np.power(np.abs(x), alpha), beta / alpha) * CONST


def scaling_form(
    orderflow_imbalance: pd.DataFrame, RN: float, QN: float, alpha: float, beta: float, CONST: float
) -> np.ndarray:
    """
    Apply the scaling form :math:`R(ΔV, T)` to a series of order flow imbalances. The scaling form
    relates the aggregate price impact `R` to the order flow imbalances `ΔV` and the time scale `T`.
    Its functional form reads:

    .. math::

        R\left(\Delta V^\prime, T\right) \cong R_T \cdot \mathscr{F}\left(\frac{\Delta V^\prime}{V_T}\right).

    where :math:`\mathcal{F}` is the scaling function, :math:`RT` and :math:`VT` are scale factors and
    :math:`\text{{CONST}}` is a constant factor in the scaling form and the orderflow imbalance ` ΔV'`.

    Args:
        orderflow_imbalance (pd.DataFrame): A DataFrame containing the columns 'imbalance' and 'T'.

    Parameters:
        RN (float): Scale factor for the return scale.
        QN (float): Scale factor for the volume scale.
        alpha (float): The alpha parameter for the scaling function.
        beta (float): The beta parameter for the scaling function.
        CONST (float): A constant factor in the scaling form.

    Returns:
        np.ndarray: An array of scaled order flow imbalances.
    """

    # Extract imbalance and T from the DataFrame
    T = orderflow_imbalance["T"].values
    imbalance = orderflow_imbalance["imbalance"].values

    # Apply the scaling function to the normalized orderflow imblance
    rescaled_imbalance = imbalance / (QN * T)
    scaled_imbalance = scaling_function(rescaled_imbalance, alpha, beta, CONST)

    return (RN * T) * scaled_imbalance * CONST


def scaling_law(
    orderflow_imbalance: pd.DataFrame, chi: float, kappa: float, alpha: float, beta: float, CONST: float
) -> np.ndarray:
    """
    Apply the scaling law :math:`R'(ΔV, chi, kappa, alpha, beta)` to series of order flow imbalances. The scaling law writes:

    .. math::

        R^\prime(\Delta V^\prime, T) = T^{\chi} \times \mathcal{F}\left(\frac{\Delta V^\prime}{T^{\kappa}}\right)},

    where :math:`\mathcal{F}` is the scaling function, :math:`\chi` and :math:`\kappa` are rescaling exponents,
    and :math:`\text{{CONST}}` is a constant factor in the scaling law.

    Args:
        orderflow_imbalance (pd.DataFrame): A DataFrame containing the columns 'imbalance' and 'T'.

    Parameters:
        chi (float): The chi rescaling exponent in the scaling law.
        kappa (float): The kappa rescaling exponent in the scaling law.
        alpha (float): The alpha parameter for the scaling function.
        beta (float): The beta parameter for the scaling function.
        CONST (float): A constant factor in the scaling law.

    Returns:
        np.ndarray: An series of scaled order flow imbalances according to the scaling law.
    """

    # Extract imbalance over some T from the DataFrame
    T = orderflow_imbalance["T"].values
    imbalance = orderflow_imbalance["imbalance"].values

    # Rescale imbalance according to kappa
    rescaled_imbalance = imbalance / np.power(T, kappa)

    # Apply the scaling function to the rescaled imbalance
    scaled_imbalance = scaling_function(rescaled_imbalance, alpha, beta, CONST)

    return np.power(T, chi) * scaled_imbalance * CONST
