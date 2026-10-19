"""
Statistical significance analysis for Bin Packing Problem experiments.
Implements paired comparisons between algorithms (GA vs PSO), Wilcoxon signed-rank tests,
paired t-tests, Holm-Bonferroni multi-testing correction, and rigorous reporting.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni step-down correction for multiple hypothesis testing.

    Adjusted p-values:
        p_adj[i] = min(1.0, max(p_adj[i-1], (m - i) * p_sorted[i]))
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.empty(m, dtype=float)
    for i in range(m):
        adjusted[i] = (m - i) * sorted_p[i]

    # Enforce monotonicity
    for i in range(1, m):
        if adjusted[i] < adjusted[i - 1]:
            adjusted[i] = adjusted[i - 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty(m, dtype=float)
    result[sorted_indices] = adjusted
    return list(result)


def compare_algorithms_per_instance(
    df_runs: pd.DataFrame,
    algo1: str = "GA",
    algo2: str = "PSO",
    metric_col: str = "best_bins",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform paired comparison between two algorithms on each instance across shared seeds.

    Tests performed:
    - Paired difference (mean, std)
    - Wilcoxon signed-rank test (or exact sign test if sample size is small)
    - Paired t-test
    - Holm-Bonferroni adjusted p-values
    """
    instances = df_runs["instance"].unique()
    records = []

    for inst in instances:
        sub = df_runs[df_runs["instance"] == inst]
        p1 = sub[sub["algorithm"] == algo1].sort_values("seed")
        p2 = sub[sub["algorithm"] == algo2].sort_values("seed")

        # Merge on seed to ensure paired comparison
        merged = pd.merge(p1, p2, on="seed", suffixes=(f"_{algo1}", f"_{algo2}"))
        v1 = merged[f"{metric_col}_{algo1}"].values
        v2 = merged[f"{metric_col}_{algo2}"].values
        n = len(v1)

        diff = v1 - v2  # Negative means algo1 used fewer bins (better)
        mean_diff = float(np.mean(diff)) if n > 0 else 0.0
        sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

        # Check if identical (all zero differences)
        if np.all(diff == 0):
            stat_w, p_w = 0.0, 1.0
            stat_t, p_t = 0.0, 1.0
            note = "Identical performance across all seeds"
        else:
            # Non-zero differences present
            try:
                res_w = stats.wilcoxon(v1, v2, alternative="two-sided")
                stat_w, p_w = float(res_w.statistic), float(res_w.pvalue)
            except Exception as e:
                stat_w, p_w = np.nan, 1.0

            try:
                res_t = stats.ttest_rel(v1, v2)
                stat_t, p_t = float(res_t.statistic), float(res_t.pvalue)
            except Exception as e:
                stat_t, p_t = np.nan, 1.0

            note = (
                f"{algo1} better ({np.sum(diff < 0)}), "
                f"{algo2} better ({np.sum(diff > 0)}), "
                f"tied ({np.sum(diff == 0)})"
            )

        records.append({
            "instance": inst,
            "n_pairs": n,
            f"mean_{algo1}": round(float(np.mean(v1)), 3),
            f"mean_{algo2}": round(float(np.mean(v2)), 3),
            "mean_diff": round(mean_diff, 3),
            "sd_diff": round(sd_diff, 3),
            "wilcoxon_stat": round(stat_w, 3) if not np.isnan(stat_w) else np.nan,
            "wilcoxon_p_raw": round(p_w, 4),
            "paired_t_stat": round(stat_t, 3) if not np.isnan(stat_t) else np.nan,
            "paired_t_p_raw": round(p_t, 4),
            "performance_note": note
        })

    df_out = pd.DataFrame(records)

    # Apply Holm-Bonferroni correction
    df_out["wilcoxon_p_adj_holm"] = [
        round(p, 4) for p in holm_bonferroni_correction(df_out["wilcoxon_p_raw"].tolist())
    ]
    df_out["paired_t_p_adj_holm"] = [
        round(p, 4) for p in holm_bonferroni_correction(df_out["paired_t_p_raw"].tolist())
    ]

    df_out["significant_wilcoxon_adj"] = df_out["wilcoxon_p_adj_holm"] < alpha
    df_out["significant_t_adj"] = df_out["paired_t_p_adj_holm"] < alpha

    return df_out


def compare_algorithms_pooled(
    df_runs: pd.DataFrame,
    algo1: str = "GA",
    algo2: str = "PSO",
    metric_col: str = "best_bins"
) -> Dict:
    """
    Perform pooled paired test across all instance-seed pairs (25 pairs total).
    Includes clear explanation of heterogeneity and test interpretation.
    """
    p1 = df_runs[df_runs["algorithm"] == algo1].sort_values(["instance", "seed"])
    p2 = df_runs[df_runs["algorithm"] == algo2].sort_values(["instance", "seed"])

    merged = pd.merge(p1, p2, on=["instance", "seed"], suffixes=(f"_{algo1}", f"_{algo2}"))
    v1 = merged[f"{metric_col}_{algo1}"].values
    v2 = merged[f"{metric_col}_{algo2}"].values
    diff = v1 - v2
    n_pairs = len(diff)

    n_algo1_better = int(np.sum(diff < 0))
    n_algo2_better = int(np.sum(diff > 0))
    n_tied = int(np.sum(diff == 0))

    if np.all(diff == 0):
        stat_w, p_w = 0.0, 1.0
        stat_t, p_t = 0.0, 1.0
    else:
        try:
            res_w = stats.wilcoxon(v1, v2, alternative="two-sided")
            stat_w, p_w = float(res_w.statistic), float(res_w.pvalue)
        except Exception:
            stat_w, p_w = np.nan, 1.0

        try:
            res_t = stats.ttest_rel(v1, v2)
            stat_t, p_t = float(res_t.statistic), float(res_t.pvalue)
        except Exception:
            stat_t, p_t = np.nan, 1.0

    return {
        "algo1": algo1,
        "algo2": algo2,
        "n_total_pairs": n_pairs,
        "mean_diff": round(float(np.mean(diff)), 4),
        "sd_diff": round(float(np.std(diff, ddof=1)), 4),
        f"{algo1}_better_count": n_algo1_better,
        f"{algo2}_better_count": n_algo2_better,
        "tied_count": n_tied,
        "wilcoxon_stat": round(stat_w, 4) if not np.isnan(stat_w) else None,
        "wilcoxon_p_value": round(p_w, 6),
        "paired_t_stat": round(stat_t, 4) if not np.isnan(stat_t) else None,
        "paired_t_p_value": round(p_t, 6),
        "interpretation": (
            f"Across {n_pairs} paired runs, {algo1} won {n_algo1_better} times, "
            f"{algo2} won {n_algo2_better} times, and {n_tied} runs tied. "
            f"Wilcoxon signed-rank test p = {p_w:.4f}. "
            "Note: Pooling across instances assumes exchangeable difference distributions."
        )
    }
