"""
Performance metrics and summary generation for BPP optimization experiments.
Computes best, mean, std, worst, runtime, validity, and percentage gap to BKS.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_gap_percent(result: float, bks: float) -> float:
    """
    Compute percentage gap to Best-Known Solution (BKS):
        gap = (result - bks) / bks * 100%
    """
    if bks <= 0:
        raise ValueError(f"BKS must be strictly positive, got {bks}.")
    return (result - bks) / bks * 100.0


def compute_summary_statistics(
    df_runs: pd.DataFrame,
    group_col: str = "instance",
    metric_col: str = "best_bins",
    bks_mapping: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Compute summary statistics per group (e.g., instance or algorithm).

    Columns:
    - best
    - mean
    - sd (sample standard deviation with ddof=1, filled with 0.0 if single run or zero variance)
    - worst
    - bks
    - mean_gap_percent
    - mean_runtime_seconds
    - all_valid
    - n_runs
    - evaluations
    """
    records = []
    for group_name, group in df_runs.groupby(group_col, sort=False):
        values = group[metric_col].values
        n_runs = len(values)
        best_val = int(np.min(values))
        mean_val = float(np.mean(values))
        worst_val = int(np.max(values))
        sd_val = float(np.std(values, ddof=1)) if n_runs > 1 else 0.0
        if np.isnan(sd_val):
            sd_val = 0.0

        runtime_mean = float(group["runtime_seconds"].mean()) if "runtime_seconds" in group.columns else 0.0
        all_valid = bool(group["valid"].all()) if "valid" in group.columns else True
        evals = int(group["evaluations"].iloc[0]) if "evaluations" in group.columns else 0

        bks = None
        gap_mean = None
        if bks_mapping and group_name in bks_mapping:
            bks = bks_mapping[group_name]
            gap_mean = compute_gap_percent(mean_val, bks)
        elif "gap_percent" in group.columns and not group["gap_percent"].isnull().all():
            gap_mean = float(group["gap_percent"].mean())

        records.append({
            group_col: group_name,
            "n_runs": n_runs,
            "bks": bks,
            "best": best_val,
            "mean": round(mean_val, 4),
            "sd": round(sd_val, 4),
            "worst": worst_val,
            "mean_gap_percent": round(gap_mean, 4) if gap_mean is not None else None,
            "mean_runtime_seconds": round(runtime_mean, 4),
            "evaluations": evals,
            "all_valid": all_valid,
        })

    return pd.DataFrame(records)
