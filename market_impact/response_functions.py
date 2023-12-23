import pandas as pd
import numpy as np

from market_impact.util.utils import normalize_conditional_impact, normalize_imbalances


def _check_conditioning_validity(conditional_variable: str):
    """Validates if the given conditional variables are supported."""
    valid_conditional_variables = ["sign", "volume"]
    if conditional_variable not in valid_conditional_variables:
        raise ValueError(
            f"Unknown imbalance column: {conditional_variable}. Expected one of {valid_conditional_variables}."
        )


def price_response(
    orderbook_states: pd.DataFrame, response_col_name: str = "R1", log_prices: bool = False, conditional: bool = False
) -> pd.DataFrame:
    """
    Computes the generalized unconditional response function R(ℓ=1) the lag-dependent
    change in price between time t and t + ℓ according to

    ..math::
        \mathcal{R}(1) \vcentcolon = \langle  \varepsilon_t \cdot ( m_{t + 1} - m_t)\vert \rangle_t,

    Where we measure price returns by taking the mean difference between the mid-price m(t)
    just before the arrival of an order event, and the mid-price m(t+1) just before the
    arrival of the next event πt of the same type.

    Args
        orderbook_states (pd.DataFrame): DataFrame containing order book states.
        Response_col_name (string): Response function column name. Default is R(ℓ=1)
        log_prices (bool): Compute log returns instead of fractional returns. Default to False.
        conditional (bool): Whether to compute the conditional R(1) or unconditional impact R(v, 1).
            Default is False.

    Returns
        data (pd.DataFrame): Orderbook states updated with generalized unconditional response function R(1).
    """
    data = orderbook_states.copy()

    # Compute returns using fractional mid-price change m(t+ℓ) - m(t) or log returns logm(t+ℓ) − logm(t)
    if log_prices:
        data["midprice_change"] = np.log(data["midprice"].shift(-1).fillna(0)) - np.log(data["midprice"])
    else:
        data["midprice_change"] = data["midprice"].diff().shift(-1).fillna(0)

    # Compute conditional or unconditional impact (default is lag-1)
    if conditional:
        # Sign already accounted for in the conditioning variable
        data[response_col_name] = data["midprice_change"]
    else:
        # Explicitly account for the sign when unconditioned
        data[response_col_name] = data["midprice_change"] * data["sign"]
    return data


def aggregate_impact(
    aggregate_features: pd.DataFrame,
    conditional_variable: str = "volume",
    conditional: bool = False,
    log_prices: bool = False,
) -> pd.DataFrame:
    """
    Computes the aggregate impact of an order, where R(ℓ) is the unconditional price impact for any lag ℓ > 0:

    ..math::
        \mathcal{R}(\ell) \vcentcolon = \langle  \varepsilon_t \cdot ( m_{t + \ell} - m_t)\vert \rangle_t,

    By extending the definition of price_response to the general case. We also compute the conditional
    aggregate impact of an order by taking the price change over the interval [t, t + T ), conditioned
    to a certain volume order-flow imbalance:

    .. math::

        R(\Delta V^\prime , T) \vcentcolon = \langle m_{t+T} - m_t\vert \Delta V^\prime \rangle.

    Note: By standard properties of equality, when $T = 1$, it follows that

    .. math::

        R(\Delta V^\prime, 1) \equiv \mathcal{R}(v = \Delta V^\prime, 1)

    and R(\Delta V^\prime, 1) simply corresponds to the impact of a single order conditioned on its volume

    .. math::

        \mathcal{R}(v, 1) \vcentcolon = \langle  \varepsilon_t \cdot ( m_{t + 1} - m_t)\vert v_t = v \rangle_t.

    A similar logic applies for the sign imbalance `Δε`, where we condition on the sign of the previous order \mathcal{R}(εt, 1).

    Args:
    - aggregate_features (pd.DataFrame): DataFrame containing orderbook features, aggregated over binning frequencies T.
    - conditional_variable (str): The variable on which to condition on (i.e., sign or volume and their imbalances).
    - conditional (bool): If True, disregards the sign of the order as conditioning. Default is False.
    - log_prices (bool): If True, uses logarithm of mid-prices for the impact computation. Default is False.

    Returns:
    - pd.DataFrame:
        conditional (True); aggregate DataFrame containing the binning frequencies T (system sizes),
        normalized imbalance and conditional aggregate impact R respectively

    Note:
    - The function assumes 'event_timestamp' to be in a format convertible to pd.Timestamp.
    - If conditional set to True, the function rescales aggregate impact `R` and imbalances `ΔV` each day by
      the corresponding values of `R(1)` and the daily volume `V_D` or average volume at best `\overline{V}_best`.
    - The conditioning variables already accounts for the sign by definition.
    """
    data = aggregate_features.copy()

    # Convert 'event_timestamp' to pd.Timestamp if not already
    if type(data["event_timestamp"].iloc[0]) != pd.Timestamp:
        data["event_timestamp"] = data["event_timestamp"].apply(lambda x: pd.Timestamp(x))

    # Compute the generalized unconditional aggregate impact R(ℓ) for ℓ > 0
    data = price_response(data, response_col_name=f"R", log_prices=log_prices, conditional=conditional)

    # Compute conditional aggregate impact R(ΔV, T)
    if conditional:
        # Check if valid conditioning variable
        _check_conditioning_validity(conditional_variable=conditional_variable)

        # Rescale observables
        data = normalize_imbalances(data)  # rescale imbalances `ΔV`
        data = normalize_conditional_impact(data)  # rescale aggregate impact `R(ΔV, T)`

        # Get system size T, sign Δε or volume imbalance ΔV, and observable R
        if conditional_variable == "sign":
            aggregate_impact_df = data[["T", "sign_imbalance", "R"]]
        else:
            aggregate_impact_df = data[["T", "volume_imbalance", "R"]]

    # FIXME: move to compute R(ℓ) in aggregate features?
    # Compute unconditional aggregate impact R(ℓ)
    else:
        # Get lags ℓ and observable R
        aggregate_impact_df = data[["T", f"R"]]
        aggregate_impact_df.rename(columns={"T": "lag"}, inplace=True)  # bin_size = data["T"].unique()[0]

    return aggregate_impact_df
