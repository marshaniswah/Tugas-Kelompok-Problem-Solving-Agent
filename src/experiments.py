"""
Master experiment runner and evaluation coordinator for BPP optimization.
Executes GA, PSO, and FFD baselines, tracks budgets, saves raw/summary CSVs,
and generates statistical and graphical outputs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .data_loader import load_instance
from .validation import validate_packing_solution
from .baseline import first_fit_decreasing
from .ga import run_genetic_algorithm
from .pso import run_particle_swarm_optimization
from .metrics import compute_gap_percent, compute_summary_statistics
from .statistics import compare_algorithms_per_instance, compare_algorithms_pooled
from .plotting import plot_mean_convergence_comparison, plot_sensitivity_analysis


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load JSON experiment configuration file."""
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def run_all_experiments(
    config: Dict,
    repo_root: Union[str, Path],
    quick_mode: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Run complete benchmark experimental suite:
    1. FFD Baseline evaluation.
    2. Main GA and PSO comparison (5 instances x 5 seeds = 25 runs each).
    3. GA operator variation comparison (mutation 0.20 vs 0.05).
    4. PSO parameter sensitivity analysis (w in [0.4, 0.7, 0.9], and c1/c2 variations).
    5. Statistical testing and convergence plots.
    """
    root = Path(repo_root)
    data_dir = root / config["benchmark"]["data_dir"]
    raw_dir = root / config["paths"]["raw_dir"]
    summaries_dir = root / config["paths"]["summaries_dir"]
    figures_dir = root / config["paths"]["figures_dir"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    instances_meta = config["benchmark"]["instances"]
    bks_map = {item["filename"]: item["bks"] for item in instances_meta}

    if quick_mode:
        seeds = config["experimental_protocol"]["quick_seeds"]
        eval_budget = config["experimental_protocol"]["quick_evaluation_budget"]
        print(f"[QUICK MODE] Seeds: {seeds}, Budget: {eval_budget}")
    else:
        seeds = config["experimental_protocol"]["seeds"]
        eval_budget = config["experimental_protocol"]["evaluation_budget"]
        print(f"[FULL MODE] Seeds: {seeds}, Budget: {eval_budget}")

    ga_params = config["algorithms"]["ga"]
    pso_params = config["algorithms"]["pso"]

    # -------------------------------------------------------------
    # 1. FFD Baseline
    # -------------------------------------------------------------
    print("\n--- Running FFD Baseline ---")
    ffd_records = []
    for item in instances_meta:
        fname = item["filename"]
        bks = item["bks"]
        inst_data = load_instance(data_dir / fname)
        res_ffd = first_fit_decreasing(
            weights=inst_data["weights"],
            capacity=inst_data["capacity"],
            bks=bks
        )
        ffd_records.append({
            "algorithm": "FFD",
            "instance": fname,
            "seed": None,
            "best_bins": res_ffd["best_bins"],
            "bks": bks,
            "gap_percent": res_ffd["gap_percent"],
            "evaluations": res_ffd["evaluations"],
            "runtime_seconds": res_ffd["runtime_seconds"],
            "valid": res_ffd["valid"]
        })
        print(f"FFD on {fname}: bins={res_ffd['best_bins']} | BKS={bks} | gap={res_ffd['gap_percent']:.2f}%")

    df_ffd = pd.DataFrame(ffd_records)
    df_ffd.to_csv(raw_dir / "ffd_raw_runs.csv", index=False)

    # -------------------------------------------------------------
    # 2. Main Comparison: GA & PSO
    # -------------------------------------------------------------
    print("\n--- Running Main GA & PSO Experiments ---")
    ga_records = []
    pso_records = []
    ga_histories_by_instance = {item["filename"]: [] for item in instances_meta}
    ga_checkpoints_by_instance = {}
    pso_histories_by_instance = {item["filename"]: [] for item in instances_meta}

    for item in instances_meta:
        fname = item["filename"]
        bks = item["bks"]
        inst_data = load_instance(data_dir / fname)
        w = inst_data["weights"]
        c = inst_data["capacity"]

        print(f"\nInstance: {fname} (BKS={bks}, Items={inst_data['n_items']}, Capacity={c})")

        # GA runs
        for seed in seeds:
            res_ga = run_genetic_algorithm(
                weights=w,
                capacity=c,
                population_size=ga_params["population_size"],
                tournament_size=ga_params["tournament_size"],
                mutation_rate=ga_params["mutation_rate"],
                crossover_rate=ga_params["crossover_rate"],
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks
            )
            ga_records.append({
                "algorithm": "GA",
                "instance": fname,
                "seed": seed,
                "best_bins": res_ga["best_fitness"],
                "bks": bks,
                "gap_percent": res_ga["gap_percent"],
                "evaluations": res_ga["evaluations"],
                "runtime_seconds": res_ga["runtime_seconds"],
                "valid": res_ga["valid"]
            })
            ga_histories_by_instance[fname].append(res_ga["convergence"])
            ga_checkpoints_by_instance[fname] = res_ga["eval_checkpoints"]
            print(f"  [GA]  Seed {seed:2d}: bins={res_ga['best_fitness']} | valid={res_ga['valid']} | time={res_ga['runtime_seconds']:.2f}s")

        # PSO runs
        for seed in seeds:
            res_pso = run_particle_swarm_optimization(
                weights=w,
                capacity=c,
                population_size=pso_params["population_size"],
                w=pso_params["inertia_weight_w"],
                c1=pso_params["cognitive_coeff_c1"],
                c2=pso_params["social_coeff_c2"],
                v_min=pso_params["velocity_min"],
                v_max=pso_params["velocity_max"],
                pos_min=pso_params["position_min"],
                pos_max=pso_params["position_max"],
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks
            )
            pso_records.append({
                "algorithm": "PSO",
                "instance": fname,
                "seed": seed,
                "best_bins": res_pso["best_fitness"],
                "bks": bks,
                "gap_percent": res_pso["gap_percent"],
                "evaluations": res_pso["evaluations"],
                "runtime_seconds": res_pso["runtime_seconds"],
                "valid": res_pso["valid"]
            })
            pso_histories_by_instance[fname].append(res_pso["history"])
            print(f"  [PSO] Seed {seed:2d}: bins={res_pso['best_fitness']} | valid={res_pso['valid']} | time={res_pso['runtime_seconds']:.2f}s")

    df_ga = pd.DataFrame(ga_records)
    df_pso = pd.DataFrame(pso_records)

    df_ga.to_csv(raw_dir / "ga_raw_runs.csv", index=False)
    df_pso.to_csv(raw_dir / "pso_raw_runs.csv", index=False)

    df_all_runs = pd.concat([df_ga, df_pso, df_ffd], ignore_index=True)
    df_all_runs.to_csv(raw_dir / "all_runs_comparison.csv", index=False)

    # -------------------------------------------------------------
    # 3. Save Convergence Checkpoints Data
    # -------------------------------------------------------------
    conv_records = []
    for fname in instances_meta:
        inst_name = fname["filename"]
        ckpt = ga_checkpoints_by_instance[inst_name]
        for s_idx, s in enumerate(seeds):
            ga_curve = ga_histories_by_instance[inst_name][s_idx]
            for ev, val in zip(ckpt, ga_curve):
                conv_records.append({
                    "algorithm": "GA",
                    "instance": inst_name,
                    "seed": s,
                    "evaluations": ev,
                    "best_so_far": val
                })
            pso_curve = pso_histories_by_instance[inst_name][s_idx]
            # Downsample PSO to checkpoints for manageable CSV size
            for ev in ckpt:
                pso_val = pso_curve[min(ev - 1, len(pso_curve) - 1)]
                conv_records.append({
                    "algorithm": "PSO",
                    "instance": inst_name,
                    "seed": s,
                    "evaluations": ev,
                    "best_so_far": pso_val
                })

    df_convergence = pd.DataFrame(conv_records)
    df_convergence.to_csv(raw_dir / "convergence_checkpoints.csv", index=False)

    # -------------------------------------------------------------
    # 4. Generate Convergence Figures
    # -------------------------------------------------------------
    print("\n--- Generating Convergence Curves ---")
    for item in instances_meta:
        fname = item["filename"]
        bks = item["bks"]
        plot_path = figures_dir / f"convergence_{Path(fname).stem}.png"
        plot_mean_convergence_comparison(
            ga_histories=ga_histories_by_instance[fname],
            ga_checkpoints=ga_checkpoints_by_instance[fname],
            pso_histories=pso_histories_by_instance[fname],
            instance_name=fname,
            bks=bks,
            output_path=plot_path
        )
        print(f"Saved figure: {plot_path.name}")

    # -------------------------------------------------------------
    # 5. GA Operator Variation Experiment
    # -------------------------------------------------------------
    print("\n--- Running GA Operator Variation Experiment ---")
    var_conf = config.get("ga_variations", {})
    var_inst = var_conf.get("target_instance", "Falkenauer_u120_00.txt")
    inst_data = load_instance(data_dir / var_inst)
    var_records = []

    for cfg in var_conf.get("configurations", []):
        cfg_name = cfg["name"]
        mut_rate = cfg["mutation_rate"]
        cr_rate = cfg["crossover_rate"]
        print(f"Testing variation: {cfg_name} (mutation_rate={mut_rate}, crossover_rate={cr_rate})")

        for seed in seeds:
            res_var = run_genetic_algorithm(
                weights=inst_data["weights"],
                capacity=inst_data["capacity"],
                population_size=ga_params["population_size"],
                tournament_size=ga_params["tournament_size"],
                mutation_rate=mut_rate,
                crossover_rate=cr_rate,
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks_map[var_inst]
            )
            var_records.append({
                "configuration": cfg_name,
                "instance": var_inst,
                "mutation_rate": mut_rate,
                "crossover_rate": cr_rate,
                "seed": seed,
                "best_bins": res_var["best_fitness"],
                "gap_percent": res_var["gap_percent"],
                "evaluations": res_var["evaluations"],
                "runtime_seconds": res_var["runtime_seconds"],
                "valid": res_var["valid"]
            })

    df_ga_variation = pd.DataFrame(var_records)
    df_ga_variation.to_csv(raw_dir / "ga_variation_raw.csv", index=False)

    df_ga_var_summary = compute_summary_statistics(
        df_ga_variation, group_col="configuration", metric_col="best_bins"
    )
    df_ga_var_summary.to_csv(summaries_dir / "ga_variation_summary.csv", index=False)

    # -------------------------------------------------------------
    # 6. PSO Sensitivity Analysis
    # -------------------------------------------------------------
    print("\n--- Running PSO Parameter Sensitivity Analysis ---")
    sens_conf = config.get("sensitivity_analysis", {})
    sens_inst = sens_conf.get("pilot_instance", "Falkenauer_u120_04.txt")
    sens_inst_data = load_instance(data_dir / sens_inst)
    bks_pilot = bks_map[sens_inst]

    # Test w values
    pso_w_records = []
    for w_val in sens_conf.get("pso_w_values", [0.4, 0.7, 0.9]):
        for seed in seeds:
            res_pso_sens = run_particle_swarm_optimization(
                weights=sens_inst_data["weights"],
                capacity=sens_inst_data["capacity"],
                population_size=pso_params["population_size"],
                w=w_val,
                c1=pso_params["cognitive_coeff_c1"],
                c2=pso_params["social_coeff_c2"],
                v_min=pso_params["velocity_min"],
                v_max=pso_params["velocity_max"],
                pos_min=pso_params["position_min"],
                pos_max=pso_params["position_max"],
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks_pilot
            )
            pso_w_records.append({
                "instance": sens_inst,
                "w": w_val,
                "seed": seed,
                "best_bins": res_pso_sens["best_fitness"],
                "valid": res_pso_sens["valid"],
                "evaluations": res_pso_sens["evaluations"],
                "runtime_seconds": res_pso_sens["runtime_seconds"],
            })

    df_pso_w = pd.DataFrame(pso_w_records)
    df_pso_w.to_csv(raw_dir / "pso_sensitivity_w_raw.csv", index=False)

    df_pso_w_summary = compute_summary_statistics(
        df_pso_w, group_col="w", metric_col="best_bins"
    )
    df_pso_w_summary.to_csv(summaries_dir / "pso_sensitivity_w_summary.csv", index=False)

    # Test C1 / C2 combinations
    pso_c_records = []
    for c_cfg in sens_conf.get("pso_c1_c2_configs", []):
        cfg_name = c_cfg["config"]
        c1_val = c_cfg["c1"]
        c2_val = c_cfg["c2"]
        for seed in seeds:
            res_pso_c = run_particle_swarm_optimization(
                weights=sens_inst_data["weights"],
                capacity=sens_inst_data["capacity"],
                population_size=pso_params["population_size"],
                w=0.7,
                c1=c1_val,
                c2=c2_val,
                v_min=pso_params["velocity_min"],
                v_max=pso_params["velocity_max"],
                pos_min=pso_params["position_min"],
                pos_max=pso_params["position_max"],
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks_pilot
            )
            pso_c_records.append({
                "instance": sens_inst,
                "config": cfg_name,
                "c1": c1_val,
                "c2": c2_val,
                "seed": seed,
                "best_bins": res_pso_c["best_fitness"],
                "valid": res_pso_c["valid"],
                "evaluations": res_pso_c["evaluations"],
                "runtime_seconds": res_pso_c["runtime_seconds"],
            })

    df_pso_c = pd.DataFrame(pso_c_records)
    df_pso_c.to_csv(raw_dir / "pso_sensitivity_c1c2_raw.csv", index=False)

    df_pso_c_summary = compute_summary_statistics(
        df_pso_c, group_col="config", metric_col="best_bins"
    )
    df_pso_c_summary.to_csv(summaries_dir / "pso_sensitivity_c1c2_summary.csv", index=False)

    # -------------------------------------------------------------
    # 7. GA Parameter Sensitivity Analysis
    # -------------------------------------------------------------
    print("\n--- Running GA Parameter Sensitivity Analysis ---")
    ga_sens_summaries = run_ga_sensitivity_analysis(
        config=config,
        repo_root=root,
        quick_mode=quick_mode
    )

    # -------------------------------------------------------------
    # 8. Statistical Analysis & Comparison Summaries
    # -------------------------------------------------------------
    print("\n--- Running Statistical Significance Analysis ---")
    df_stat_instance = compare_algorithms_per_instance(
        df_runs=df_all_runs,
        algo1="GA",
        algo2="PSO",
        metric_col="best_bins",
        alpha=config.get("statistical_testing", {}).get("alpha", 0.05)
    )
    df_stat_instance.to_csv(summaries_dir / "statistical_tests_per_instance.csv", index=False)

    pooled_stat = compare_algorithms_pooled(
        df_runs=df_all_runs,
        algo1="GA",
        algo2="PSO",
        metric_col="best_bins"
    )
    with open(summaries_dir / "statistical_test_pooled.json", "w", encoding="utf-8") as f:
        json.dump(pooled_stat, f, indent=2)

    # Overall comparison summary table
    summary_records = []
    for algo in ["FFD", "GA", "PSO"]:
        sub = df_all_runs[df_all_runs["algorithm"] == algo]
        summary = compute_summary_statistics(
            sub, group_col="instance", metric_col="best_bins", bks_mapping=bks_map
        )
        summary.insert(0, "algorithm", algo)
        summary_records.append(summary)

    df_comparison_summary = pd.concat(summary_records, ignore_index=True)
    df_comparison_summary.to_csv(summaries_dir / "bpp_comparison_summary.csv", index=False)

    # Save Markdown tables in reports/tables/
    reports_dir = root / config["paths"]["reports_tables_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "bpp_comparison_summary.md", "w", encoding="utf-8") as f:
        f.write(df_comparison_summary.to_markdown(index=False))

    with open(reports_dir / "statistical_tests_per_instance.md", "w", encoding="utf-8") as f:
        f.write(df_stat_instance.to_markdown(index=False))

    with open(reports_dir / "pso_sensitivity_w.md", "w", encoding="utf-8") as f:
        f.write(df_pso_w_summary.to_markdown(index=False))

    with open(reports_dir / "pso_sensitivity_c1c2.md", "w", encoding="utf-8") as f:
        f.write(df_pso_c_summary.to_markdown(index=False))

    with open(reports_dir / "ga_variation.md", "w", encoding="utf-8") as f:
        f.write(df_ga_var_summary.to_markdown(index=False))

    print("\n=== Experiment Suite Successfully Completed ===")
    return {
        "all_runs": df_all_runs,
        "summary": df_comparison_summary,
        "statistics": df_stat_instance,
        "ga_sensitivity": ga_sens_summaries
    }


def run_ga_sensitivity_analysis(
    config: Dict,
    repo_root: Union[str, Path],
    quick_mode: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Run systematic GA parameter sensitivity experiments on pilot instance:
    1. Population Size: N in [20, 30, 40, 50, 60] (with pm=0.20, k=3, pc=0.90)
    2. Mutation Rate: pm in [0.01, 0.05, 0.10, 0.20] (with N=50, k=3, pc=0.90)
    3. Tournament Size: k in [2, 3, 4, 5] (with N=50, pm=0.20, pc=0.90)
    4. Crossover Rate: pc in [0.70, 0.80, 0.90, 1.00] (with N=50, pm=0.20, k=3)

    Generates raw CSVs, summary CSVs, and markdown report tables.
    """
    root = Path(repo_root)
    data_dir = root / config["benchmark"]["data_dir"]
    raw_dir = root / config["paths"]["raw_dir"]
    summaries_dir = root / config["paths"]["summaries_dir"]
    reports_dir = root / config["paths"]["reports_tables_dir"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ga_sens_conf = config.get("ga_sensitivity_analysis", {})
    pilot_inst = ga_sens_conf.get("pilot_instance", "Falkenauer_u120_00.txt")
    inst_data = load_instance(data_dir / pilot_inst)
    weights = inst_data["weights"]
    capacity = inst_data["capacity"]

    instances_meta = config["benchmark"]["instances"]
    bks_map = {item["filename"]: item["bks"] for item in instances_meta}
    bks = bks_map.get(pilot_inst, 48)

    if quick_mode:
        seeds = config["experimental_protocol"]["quick_seeds"]
        eval_budget = config["experimental_protocol"]["quick_evaluation_budget"]
    else:
        seeds = config["experimental_protocol"]["seeds"]
        eval_budget = config["experimental_protocol"]["evaluation_budget"]

    # Baseline GA parameters used during single-parameter isolation
    base_pop = 50
    base_tour = 3
    base_pm = 0.20
    base_pc = 0.90

    # 1. Population Size Tuning
    pop_sizes = ga_sens_conf.get("population_sizes", [20, 30, 40, 50, 60])
    pop_records = []
    print(f"[GA Sensitivity] Testing Population Sizes: {pop_sizes}...")
    for p in pop_sizes:
        for seed in seeds:
            res = run_genetic_algorithm(
                weights=weights,
                capacity=capacity,
                population_size=p,
                tournament_size=base_tour,
                mutation_rate=base_pm,
                crossover_rate=base_pc,
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks
            )
            pop_records.append({
                "population_size": p,
                "seed": seed,
                "best_bins": res["best_fitness"],
                "gap_percent": res["gap_percent"],
                "evaluations": res["evaluations"],
                "runtime_seconds": res["runtime_seconds"],
                "valid": res["valid"]
            })
    df_pop_raw = pd.DataFrame(pop_records)
    df_pop_raw.to_csv(raw_dir / "ga_sensitivity_population_raw.csv", index=False)
    df_pop_summary = compute_summary_statistics(df_pop_raw, group_col="population_size", metric_col="best_bins")
    df_pop_summary.to_csv(summaries_dir / "ga_sensitivity_population_summary.csv", index=False)
    with open(reports_dir / "ga_sensitivity_population.md", "w", encoding="utf-8") as f:
        f.write(df_pop_summary.to_markdown(index=False))

    # 2. Mutation Rate Tuning
    mut_rates = ga_sens_conf.get("mutation_rates", [0.01, 0.05, 0.10, 0.20])
    mut_records = []
    print(f"[GA Sensitivity] Testing Mutation Rates: {mut_rates}...")
    for m in mut_rates:
        for seed in seeds:
            res = run_genetic_algorithm(
                weights=weights,
                capacity=capacity,
                population_size=base_pop,
                tournament_size=base_tour,
                mutation_rate=m,
                crossover_rate=base_pc,
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks
            )
            mut_records.append({
                "mutation_rate": m,
                "seed": seed,
                "best_bins": res["best_fitness"],
                "gap_percent": res["gap_percent"],
                "evaluations": res["evaluations"],
                "runtime_seconds": res["runtime_seconds"],
                "valid": res["valid"]
            })
    df_mut_raw = pd.DataFrame(mut_records)
    df_mut_raw.to_csv(raw_dir / "ga_sensitivity_mutation_raw.csv", index=False)
    df_mut_summary = compute_summary_statistics(df_mut_raw, group_col="mutation_rate", metric_col="best_bins")
    df_mut_summary.to_csv(summaries_dir / "ga_sensitivity_mutation_summary.csv", index=False)
    with open(reports_dir / "ga_sensitivity_mutation.md", "w", encoding="utf-8") as f:
        f.write(df_mut_summary.to_markdown(index=False))

    # 3. Tournament Size Tuning
    tour_sizes = ga_sens_conf.get("tournament_sizes", [2, 3, 4, 5])
    tour_records = []
    print(f"[GA Sensitivity] Testing Tournament Sizes: {tour_sizes}...")
    for t in tour_sizes:
        for seed in seeds:
            res = run_genetic_algorithm(
                weights=weights,
                capacity=capacity,
                population_size=base_pop,
                tournament_size=t,
                mutation_rate=base_pm,
                crossover_rate=base_pc,
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks
            )
            tour_records.append({
                "tournament_size": t,
                "seed": seed,
                "best_bins": res["best_fitness"],
                "gap_percent": res["gap_percent"],
                "evaluations": res["evaluations"],
                "runtime_seconds": res["runtime_seconds"],
                "valid": res["valid"]
            })
    df_tour_raw = pd.DataFrame(tour_records)
    df_tour_raw.to_csv(raw_dir / "ga_sensitivity_tournament_raw.csv", index=False)
    df_tour_summary = compute_summary_statistics(df_tour_raw, group_col="tournament_size", metric_col="best_bins")
    df_tour_summary.to_csv(summaries_dir / "ga_sensitivity_tournament_summary.csv", index=False)
    with open(reports_dir / "ga_sensitivity_tournament.md", "w", encoding="utf-8") as f:
        f.write(df_tour_summary.to_markdown(index=False))

    # 4. Crossover Rate Tuning
    cr_rates = ga_sens_conf.get("crossover_rates", [0.70, 0.80, 0.90, 1.00])
    cr_records = []
    print(f"[GA Sensitivity] Testing Crossover Rates: {cr_rates}...")
    for cr in cr_rates:
        for seed in seeds:
            res = run_genetic_algorithm(
                weights=weights,
                capacity=capacity,
                population_size=base_pop,
                tournament_size=base_tour,
                mutation_rate=base_pm,
                crossover_rate=cr,
                evaluation_budget=eval_budget,
                seed=seed,
                bks=bks
            )
            cr_records.append({
                "crossover_rate": cr,
                "seed": seed,
                "best_bins": res["best_fitness"],
                "gap_percent": res["gap_percent"],
                "evaluations": res["evaluations"],
                "runtime_seconds": res["runtime_seconds"],
                "valid": res["valid"]
            })
    df_cr_raw = pd.DataFrame(cr_records)
    df_cr_raw.to_csv(raw_dir / "ga_sensitivity_crossover_raw.csv", index=False)
    df_cr_summary = compute_summary_statistics(df_cr_raw, group_col="crossover_rate", metric_col="best_bins")
    df_cr_summary.to_csv(summaries_dir / "ga_sensitivity_crossover_summary.csv", index=False)
    with open(reports_dir / "ga_sensitivity_crossover.md", "w", encoding="utf-8") as f:
        f.write(df_cr_summary.to_markdown(index=False))

    return {
        "population": df_pop_summary,
        "mutation": df_mut_summary,
        "tournament": df_tour_summary,
        "crossover": df_cr_summary,
    }

