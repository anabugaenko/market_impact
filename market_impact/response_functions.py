import pandas as pd
import numpy as np

from market_impact.util.data_utils import (
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
        \\mathcal{R}(1) \vcentcolon = \\langle  \varepsilon_t \\cdot ( m_{t + 1} - m_t)\vert \rangle_t,

    Args
        orderbook_states (pd.DataFrame): DataFrame containing order book states.
        Response_col_name (string): Response function column name. Default is lag-1 impact R(ℓ=1).
        log_prices (bool): Compute log returns instead of fractional returns. Default to False.
        conditional (bool): Whether to compute the conditional or uconditional impact. Default is False.

    Returns
        data (pd.DataFrame): Orderbook states DataFrame updated with lag-1 unconditional impact R(1).
    """
    data = orderbook_states.copy()

    # Compute log or fractional returns
    if log_prices:
        data["midprice_change"] = np.log(data["midprice"].shift(-1).fillna(0)) - np.log(data["midprice"])
    else:
        data["midprice_change"] = data["midprice"].diff().shift(-1).fillna(0)

    if conditional:
        # Sign already accounted for in the conditioning variable
        data[response_col_name] = data["midprice_change"]
    else:
        # Explicitly account for the sign when unconditioned
        data[response_col_name] = data["midprice_change"] * data["sign"]

    return data


def unconditional_imapact(aggregate_features: pd.DataFrame, log_prices: bool = False):
    """
    Compute the generlaized unconditional aggregate impact of an order, where R(ℓ) is the price change for any lag ℓ > 0.

    ..math::
        \\mathcal{R}(\\ell) \vcentcolon = \\langle  \varepsilon_t \\cdot ( m_{t + \\ell} - m_t)\vert \rangle_t.

    Args:
        aggregate_features (pd.DataFrame): DataFrame containing orderbook features, aggregated over lagged frequencies ℓ.
        log_prices (bool): If True, uses logarithm of mid-prices for the impact computation. Default is False.

    Returns:
        unconditional_impact (pd.DataFrame): DataFrame containing unconditional impact ["lag", "Rℓ"].

    Note:
        The function assumes 'event_timestamp' to be in a format convertible to pd.Timestamp.
    """
    # FIXME: ensure we compute R(ℓ) on a daily basis to avoid propagation of errors
    data = aggregate_features.copy()

    # Convert 'event_timestamp' to pd.Timestamp if not already
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))

    # Compute the unconditional price impact R(ℓ)
    data = price_response(
        data, response_col_name=f"Rℓ", log_prices=log_prices, conditional=False
    )
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
    Computes the conditional aggregate impact of an order, where we calculate returns of an order by taking
    the averge price change over the time interval [t, t + T ), conditioned to a certain order-flow imbalance.

    .. math::

        R(\\Delta V^\\prime , T) \vcentcolon = \langle m_{t+T} - m_t\vert \\Delta V^\\prime \rangle.

    Args:
        aggregate_features (pd.DataFrame): DataFrame containing orderbook features, aggregated over bin frequencies T.
        log_prices (bool): If True, uses logarithm of mid-prices for the impact computation. Default is False.
        normalization_constant (str): Normalize the data by corresponding daily or average values. Default is "daily".
        conditional_variable (str): The variable on which to condition on (i.e., sign Δε or volume imbalance ΔV).

    Returns:
        aggregate_impact (pd.DataFrame): DataFrame containing aggregate impact ["T", "imbalance", "R"].

    Note:
        The conditioning variables already accounts for the sign by definition.
    """
    data = aggregate_features.copy()

    # Convert 'event_timestamp' to pd.Timestamp if not already
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))

    data = price_response(data, response_col_name=f"R", log_prices=log_prices, conditional=True)

    # Normalize R and imbalance by corresponding values
    data = normalize_aggregate_impact(data)
    data = normalize_imbalances(data, normalization_constant=normalization_constant,
        conditional_variable=conditional_variable)

    # Return R conditioned to a certain imbalance
    if conditional_variable == "sign_imbalance":
        aggregate_impact = data[["T", "sign_imbalance", "R"]]
    else:
        aggregate_impact = data[["T", "volume_imbalance", "R"]]

    return aggregate_impact
