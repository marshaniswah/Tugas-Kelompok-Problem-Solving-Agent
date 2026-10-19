"""
Plotting and visualization module for Bin Packing Problem experiments.
Generates publication-quality convergence curves and sensitivity analysis plots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set publication-quality plot style
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 13,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": True,
})


def plot_mean_convergence_comparison(
    ga_histories: List[List[float]],
    ga_checkpoints: List[int],
    pso_histories: List[List[float]],
    instance_name: str,
    bks: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    show_std: bool = True
) -> None:
    """
    Plot mean convergence curves for GA and PSO on comparable evaluation checkpoints.

    Parameters
    ----------
    ga_histories : list of list of float
        GA convergence curves across seeds (one curve per seed).
    ga_checkpoints : list of int
        Evaluation counts corresponding to each GA history checkpoint.
    pso_histories : list of list of float
        PSO convergence curves across seeds (eval 1..6000).
    instance_name : str
        Benchmark instance name.
    bks : int, optional
        Best-Known Solution / optimum line.
    output_path : str or Path, optional
        File path to save the plot.
    show_std : bool
        Whether to show shaded +/- 1 SD bands.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)

    # GA processing
    ga_arr = np.asarray(ga_histories, dtype=float)
    ga_mean = np.mean(ga_arr, axis=0)
    ga_sd = np.std(ga_arr, axis=0)

    # PSO processing
    pso_arr = np.asarray(pso_histories, dtype=float)
    pso_mean = np.mean(pso_arr, axis=0)
    pso_sd = np.std(pso_arr, axis=0)
    pso_x = np.arange(1, pso_arr.shape[1] + 1)

    # Plot PSO curve
    ax.plot(
        pso_x, pso_mean,
        label=f"PSO Mean (n={len(pso_histories)})",
        color="#1f77b4", linewidth=1.8
    )
    if show_std and pso_arr.shape[0] > 1:
        ax.fill_between(
            pso_x, pso_mean - pso_sd, pso_mean + pso_sd,
            color="#1f77b4", alpha=0.15, label="PSO ±1 SD"
        )

    # Plot GA curve (stepped or linear at evaluation checkpoints)
    ax.plot(
        ga_checkpoints, ga_mean,
        label=f"GA Mean (n={len(ga_histories)})",
        color="#d62728", linewidth=1.8, marker="o", markersize=3, markevery=5
    )
    if show_std and ga_arr.shape[0] > 1:
        ax.fill_between(
            ga_checkpoints, ga_mean - ga_sd, ga_mean + ga_sd,
            color="#d62728", alpha=0.15, label="GA ±1 SD"
        )

    # Optimum / BKS reference line
    if bks is not None:
        ax.axhline(
            y=bks, color="#2ca02c", linestyle="--", linewidth=1.5,
            label=f"Optimum / BKS ({bks})"
        )

    ax.set_title(f"Mean Convergence Comparison — {instance_name}", fontweight="bold", pad=10)
    ax.set_xlabel("Objective Function Evaluations", fontweight="medium")
    ax.set_ylabel("Best-so-far Number of Bins (Minimized)", fontweight="medium")
    ax.set_xlim(0, max(max(ga_checkpoints), len(pso_x)) + 50)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_sensitivity_analysis(
    df_sens: pd.DataFrame,
    param_col: str,
    metric_col: str = "best_bins",
    title: str = "Parameter Sensitivity Analysis",
    xlabel: str = "Parameter Value",
    ylabel: str = "Number of Bins",
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Generate parameter sensitivity plot showing mean and min/max bounds.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)

    summary = df_sens.groupby(param_col)[metric_col].agg(
        mean="mean", min="min", max="max", std="std"
    ).reset_index()

    summary["std"] = summary["std"].fillna(0)
    x = summary[param_col]

    ax.errorbar(
        x, summary["mean"], yerr=summary["std"],
        fmt="-o", color="#2b5c8f", ecolor="#6baed6", elinewidth=1.5,
        capsize=4, capthick=1.5, linewidth=2, label="Mean ± 1 SD"
    )

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontweight="medium")
    ax.set_ylabel(ylabel, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
