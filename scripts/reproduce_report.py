#!/usr/bin/env python3
"""
Report and figure reproduction script: Regenerates all summary tables,
statistical significance reports, and convergence figures directly from saved raw CSV results.
Does NOT rerun the metaheuristics, executing in seconds.

Usage:
    python scripts/reproduce_report.py --config config/experiment_config.json
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

# Ensure src module is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.experiments import load_config
from src.metrics import compute_summary_statistics
from src.statistics import compare_algorithms_per_instance, compare_algorithms_pooled
from src.plotting import plot_mean_convergence_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all BPP summary tables and figures from saved raw results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment_config.json",
        help="Path to experiment configuration JSON file"
    )
    args = parser.parse_args()

    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    raw_dir = repo_root / config["paths"]["raw_dir"]
    summaries_dir = repo_root / config["paths"]["summaries_dir"]
    figures_dir = repo_root / config["paths"]["figures_dir"]
    reports_dir = repo_root / config["paths"]["reports_tables_dir"]

    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    instances_meta = config["benchmark"]["instances"]
    bks_map = {item["filename"]: item["bks"] for item in instances_meta}

    raw_all_path = raw_dir / "all_runs_comparison.csv"
    if not raw_all_path.is_file():
        raise FileNotFoundError(
            f"Raw results file not found at: {raw_all_path}. "
            "Please run 'python scripts/run_experiments.py' first."
        )

    print(f"Loading raw runs from: {raw_all_path}")
    df_all_runs = pd.read_csv(raw_all_path)

    # 1. Regenerate Comparison Summary Table
    print("Generating comparison summary table...")
    summary_records = []
    for algo in ["FFD", "GA", "PSO"]:
        sub = df_all_runs[df_all_runs["algorithm"] == algo]
        summary = compute_summary_statistics(
            sub, group_col="instance", metric_col="best_bins", bks_mapping=bks_map
        )
        summary.insert(0, "algorithm", algo)
        summary_records.append(summary)

    df_comparison_summary = pd.concat(summary_records, ignore_index=True)
    summary_csv = summaries_dir / "bpp_comparison_summary.csv"
    df_comparison_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Save Markdown Table
    summary_md = reports_dir / "bpp_comparison_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(df_comparison_summary.to_markdown(index=False))
    print(f"Saved: {summary_md}")

    # 2. Regenerate Statistical Significance Tests
    print("Computing statistical significance tests (Wilcoxon and paired t-test)...")
    alpha = config.get("statistical_testing", {}).get("alpha", 0.05)
    df_stat_instance = compare_algorithms_per_instance(
        df_runs=df_all_runs,
        algo1="GA",
        algo2="PSO",
        metric_col="best_bins",
        alpha=alpha
    )
    stat_csv = summaries_dir / "statistical_tests_per_instance.csv"
    df_stat_instance.to_csv(stat_csv, index=False)
    print(f"Saved: {stat_csv}")

    stat_md = reports_dir / "statistical_tests_per_instance.md"
    with open(stat_md, "w", encoding="utf-8") as f:
        f.write(df_stat_instance.to_markdown(index=False))
    print(f"Saved: {stat_md}")

    pooled_stat = compare_algorithms_pooled(
        df_runs=df_all_runs,
        algo1="GA",
        algo2="PSO",
        metric_col="best_bins"
    )
    stat_json = summaries_dir / "statistical_test_pooled.json"
    with open(stat_json, "w", encoding="utf-8") as f:
        json.dump(pooled_stat, f, indent=2)
    print(f"Saved: {stat_json}")

    # 3. Regenerate Convergence Figures if convergence data is available
    conv_path = raw_dir / "convergence_checkpoints.csv"
    if conv_path.is_file():
        print(f"Regenerating convergence plots from: {conv_path}...")
        df_conv = pd.read_csv(conv_path)

        for item in instances_meta:
            fname = item["filename"]
            bks = item["bks"]
            sub_inst = df_conv[df_conv["instance"] == fname]

            # Extract GA curves
            sub_ga = sub_inst[sub_inst["algorithm"] == "GA"]
            ga_checkpoints = sorted(sub_ga["evaluations"].unique())
            ga_curves = []
            for s in sub_ga["seed"].unique():
                curve = sub_ga[sub_ga["seed"] == s].sort_values("evaluations")["best_so_far"].tolist()
                ga_curves.append(curve)

            # Extract PSO curves
            sub_pso = sub_inst[sub_inst["algorithm"] == "PSO"]
            pso_curves = []
            for s in sub_pso["seed"].unique():
                curve = sub_pso[sub_pso["seed"] == s].sort_values("evaluations")["best_so_far"].tolist()
                pso_curves.append(curve)

            fig_path = figures_dir / f"convergence_{Path(fname).stem}.png"
            plot_mean_convergence_comparison(
                ga_histories=ga_curves,
                ga_checkpoints=ga_checkpoints,
                pso_histories=pso_curves,
                instance_name=fname,
                bks=bks,
                output_path=fig_path
            )
            print(f"Regenerated figure: {fig_path.name}")

    # 4. Regenerate GA Sensitivity Summaries & Reports
    col_map = {
        "population": "population_size",
        "mutation": "mutation_rate",
        "tournament": "tournament_size",
        "crossover": "crossover_rate"
    }
    for param_name, grp_col in col_map.items():
        raw_p = raw_dir / f"ga_sensitivity_{param_name}_raw.csv"
        if raw_p.is_file():
            print(f"Regenerating GA sensitivity table for: {param_name}...")
            df_param_raw = pd.read_csv(raw_p)
            df_param_summary = compute_summary_statistics(df_param_raw, group_col=grp_col, metric_col="best_bins")
            sum_csv = summaries_dir / f"ga_sensitivity_{param_name}_summary.csv"
            df_param_summary.to_csv(sum_csv, index=False)
            sum_md = reports_dir / f"ga_sensitivity_{param_name}.md"
            with open(sum_md, "w", encoding="utf-8") as f:
                f.write(df_param_summary.to_markdown(index=False))
            print(f"Saved: {sum_md}")

    # 5. Regenerate PSO Sensitivity Summaries & Reports
    pso_w_raw = raw_dir / "pso_sensitivity_w_raw.csv"
    if pso_w_raw.is_file():
        print("Regenerating PSO sensitivity (w)...")
        df_pso_w = pd.read_csv(pso_w_raw)
        df_pso_w_sum = compute_summary_statistics(df_pso_w, group_col="w", metric_col="best_bins")
        df_pso_w_sum.to_csv(summaries_dir / "pso_sensitivity_w_summary.csv", index=False)
        with open(reports_dir / "pso_sensitivity_w.md", "w", encoding="utf-8") as f:
            f.write(df_pso_w_sum.to_markdown(index=False))

    pso_c_raw = raw_dir / "pso_sensitivity_c1c2_raw.csv"
    if pso_c_raw.is_file():
        print("Regenerating PSO sensitivity (c1/c2)...")
        df_pso_c = pd.read_csv(pso_c_raw)
        df_pso_c_sum = compute_summary_statistics(df_pso_c, group_col="config", metric_col="best_bins")
        df_pso_c_sum.to_csv(summaries_dir / "pso_sensitivity_c1c2_summary.csv", index=False)
        with open(reports_dir / "pso_sensitivity_c1c2.md", "w", encoding="utf-8") as f:
            f.write(df_pso_c_sum.to_markdown(index=False))

    ga_var_raw = raw_dir / "ga_variation_raw.csv"
    if ga_var_raw.is_file():
        print("Regenerating GA variation...")
        df_ga_var = pd.read_csv(ga_var_raw)
        df_ga_var_sum = compute_summary_statistics(df_ga_var, group_col="configuration", metric_col="best_bins")
        df_ga_var_sum.to_csv(summaries_dir / "ga_variation_summary.csv", index=False)
        with open(reports_dir / "ga_variation.md", "w", encoding="utf-8") as f:
            f.write(df_ga_var_sum.to_markdown(index=False))

    print("\n=== All Report Tables and Figures Successfully Regenerated ===")


if __name__ == "__main__":
    main()
