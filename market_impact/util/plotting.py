import pylab
import pandas as pd
from matplotlib import rc
from typing import List, Dict
from matplotlib import pyplot as plt

from market_impact.util.utils import bin_data_into_quantiles
from market_impact.functional_form import (
    scaling_function,
    scaling_form,
    scaling_law,
)


# Plotting
pylab.rcParams["xtick.major.pad"] = "8"
pylab.rcParams["ytick.major.pad"] = "8"
rc("text", usetex=True)
rc("mathtext", fontset="stix")
rc("axes", labelsize="large")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage{lmodern} \usepackage{amssymb}"


def plot_scaling_form(
    smoothed_data: pd.DataFrame,
    fit_parameters: Dict,
    q: int,
    x_col="volume_imbalance",
):
    """
    Plots the scaling form where the scaling function is plotted for each
    system size `T` before the transformation/renormalization.
    """
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.gca()

    for T, param in fit_parameters.items():
        params = fit_parameters[T]
        data = smoothed_data[smoothed_data["T"] == T][[x_col, "T", "R"]]
        binned_data = bin_data_into_quantiles(
            data, q=q, duplicates="drop", x_col=x_col
        )

        T_values = binned_data["T"].values
        imbalance_values = binned_data[x_col].values
        R_values = binned_data["R"].values

        # Orderflow imbalance
        orderflow_imbalance = pd.DataFrame(
            {"T": T_values, "imbalance": imbalance_values}
        )

        # Compute the model prediction
        model_predictions = scaling_form(orderflow_imbalance, *params)

        # Plotting
        plt.scatter(imbalance_values, R_values, label=f"T = {T}", s=20)
        plt.plot(imbalance_values, model_predictions)

        ax.minorticks_off()
        if x_col == "volume_imbalance":
            xlabel = r"$\it{\Delta V^\prime/V_{D}}$"
            ylabel = r"$\it{R(\Delta V^\prime,T)/\mathcal{R}(1)}$"
        elif x_col == "sign_imbalance":
            xlabel = r"$\it{\Delta \mathcal{E}^\prime/\mathcal{E}_{D}}$"
            ylabel = r"$\it{R(\Delta \mathcal{E}^\prime,T)/\mathcal{R}(1)}$"
        else:
            xlabel = x_col
            ylabel = r"\it{R}"
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        ax.tick_params(
            bottom=True,
            top=True,
            left=True,
            right=True,
            direction="in",
            width=0.5,
            size=3,
        )

    legend = ax.legend(markerfirst=True)
    for indx, T in enumerate(fit_parameters.keys()):
        legend.get_texts()[indx].set_text(f"$T = {int(T)}$")

    legend.markerscale = 0.1


def plot_scaling_law(
    rescaled_parameters: Dict, q: int, x_col="volume_imbalance"
):
    """
    Plots the scaling law where the scaling functions collapses onto a single
    master curve after the transformation/renormalization.
    """
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.gca()

    for T, rescaled_param in rescaled_parameters.items():
        params = rescaled_param.params
        data = rescaled_param.data
        binned_data = bin_data_into_quantiles(
            data, q=q, duplicates="drop", x_col=x_col
        )

        T_values = binned_data["T"].values
        imbalance_values = binned_data[x_col].values
        R_values = binned_data["R"].values

        # Orderflow imblance
        orderflow_imbalance = pd.DataFrame(
            {"T": T_values, "imbalance": imbalance_values}
        )

        # Compute the model prediction
        model_predictions = scaling_law(orderflow_imbalance, *params)

        # Plotting
        plt.scatter(imbalance_values, R_values, label=f"T = {T}", s=20)

        ax.minorticks_off()
        if x_col == "volume_imbalance":
            xlabel = r"$\it{\Delta V^\prime/V_{D}T^{\varkappa}}$"
            ylabel = r"$\it{R(\Delta V^\prime,T)/\mathcal{R}(1)T^{\chi}}$"
        elif x_col == "sign_imbalance":
            xlabel = (
                r"$\it{\Delta \mathcal{E}^\prime/\mathcal{E}_{D}T^{\varkappa}}$"
            )
            ylabel = (
                r"$\it{R(\Delta \mathcal{E}^\prime,T)/\mathcal{R}(1)T^{\chi}}$"
            )
        else:
            xlabel = x_col
            ylabel = r"\it{R}"
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        ax.tick_params(
            bottom=True,
            top=True,
            left=True,
            right=True,
            direction="in",
            width=0.5,
            size=3,
        )

    legend = ax.legend(markerfirst=True)
    for indx, T in enumerate(rescaled_parameters.keys()):
        legend.get_texts()[indx].set_text(f"$T = {int(T)}$")

    legend.markerscale = 0.1
