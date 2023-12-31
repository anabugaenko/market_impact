import pandas as pd
import numpy as np

from market_impact.util.utils import (
    normalize_imbalances,
    normalize_aggregate_impact,
)


def price_response(
    orderbook_states: pd.DataFrame,
    response_col_name: str = "R1",
    log_prices: bool = False,
    conditional: bool = False,
) -> pd.DataFrame:
    """
    Compute the price response as the lag-dependent unconditional change in mid-price m(t) between time t and t + 1.

    ..math::
        \mathcal{R}(1) \vcentcolon = \langle  \varepsilon_t \cdot ( m_{t + 1} - m_t)\vert \rangle_t,

    Args
        orderbook_states (pd.DataFrame): DataFrame containing order book states.
        Response_col_name (string): Response function column name. Default is lag-1 impact R(ℓ=1).
        log_prices (bool): Compute log returns instead of fractional returns. Default to False.
        conditional (bool): Whether to compute the conditional or uconditional impact. Default is False.

    Returns
        data (pd.DataFrame): Orderbook states DataFrame updated with lag-1 unconditional response function R(1).
    """
    data = orderbook_states.copy()

    # Compute returns
    if log_prices:
        # Log returns logm(t+1) − logm(t)
        data["midprice_change"] = np.log(
            data["midprice"].shift(-1).fillna(0)
        ) - np.log(data["midprice"])
    else:
        # Fractional mid-price change m(t+1) - m(t)
        data["midprice_change"] = data["midprice"].diff().shift(-1).fillna(0)

    # Compute conditional or unconditional impact
    if conditional:
        # Sign already accounted for in the conditioning variable
        data[response_col_name] = data["midprice_change"]
    else:
        # Explicitly account for the sign when unconditioned
        data[response_col_name] = data["midprice_change"] * data["sign"]

    return data


def unconditional_imapact(
    aggregate_features: pd.DataFrame, log_prices: bool = False
):
    """
    Compute the generlaized unconditional aggregate impact of an order, where R(ℓ) is the price response for any lag ℓ > 0.

    ..math::
        \mathcal{R}(\ell) \vcentcolon = \langle  \varepsilon_t \cdot ( m_{t + \ell} - m_t)\vert \rangle_t.

    Args:
        aggregate_features (pd.DataFrame): DataFrame containing orderbook features, aggregated over lagged frequencies ℓ.
        log_prices (bool): If True, uses logarithm of mid-prices for the impact computation. Default is False.

    Returns:
        unconditional_aggregate_impact (pd.DataFrame): DataFrame containing unconditional impact ["lag", "Rℓ"]

    Note:
        The function assumes 'event_timestamp' to be in a format convertible to pd.Timestamp.
    """
    data = aggregate_features.copy()

    # Convert 'event_timestamp' to pd.Timestamp if not already
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(
            lambda x: pd.Timestamp(x)
        )

    # Compute the unconditional price impact R(ℓ)
    data = price_response(
        data, response_col_name=f"Rℓ", log_prices=log_prices, conditional=False
    )

    # Get lag-ℓ and correpsonding price impact
    unconditional_impact = data[["T", "Rℓ"]]
    unconditional_impact.rename(columns={"T": "lag"}, inplace=True)

    return unconditional_impact


def aggregate_impact(
    aggregate_features: pd.DataFrame,
    log_prices: bool = False,
    normalization_constant: str = "daily",
    conditional_variable: str = "volume_imbalance",
) -> pd.DataFrame:
    """
    Computes the aggregate impact of an order, where  we consider the conditional aggregate impact of an order
    by taking the averge price change over the interval [t, t + T ), conditioned to a certain order-flow imbalance.

    .. math::

        R(\Delta V^\prime , T) \vcentcolon = \langle m_{t+T} - m_t\vert \Delta V^\prime \rangle.

    Args:
        aggregate_features (pd.DataFrame): DataFrame containing orderbook features, aggregated over binning frequencies T.
        log_prices (bool): If True, uses logarithm of mid-prices for the impact computation. Default is False.
        normalization_constant (str): Whether to normalize the data by correpsonding daily or average values. Default is "daily".
        conditional_variable (str): The variable on which to condition on (i.e., sign imbalance Δε or volume imbalance ΔV).

    Returns:
        conditonal_aggregate_impact (pd.DataFrame): DataFrame containing conditional aggregate impact ["T", "imbalance", "R"]

    Note:
        The conditioning variables already accounts for the sign by definition.
    """
    data = aggregate_features.copy()

    # Convert 'event_timestamp' to pd.Timestamp if not already
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(
            lambda x: pd.Timestamp(x)
        )

    # Compute the conditional aggregate impact R(ΔV, T)
    data = price_response(
        data, response_col_name=f"R", log_prices=log_prices, conditional=True
    )

    # Normalize price impact and imbalance each day by the corresponding values
    data = normalize_aggregate_impact(data)
    data = normalize_imbalances(
        data,
        normalization_constant=normalization_constant,
        conditional_variable=conditional_variable,
    )

    # Get system sizes T, imbalance Δ, and observable R
    if conditional_variable == "sign_imbalance":
        aggregate_impact = data[["T", "sign_imbalance", "R"]]
    else:
        aggregate_impact = data[["T", "volume_imbalance", "R"]]

    return aggregate_impact
