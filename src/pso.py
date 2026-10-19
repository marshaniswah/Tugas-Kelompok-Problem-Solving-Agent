"""
Custom Particle Swarm Optimization (PSO) for the 1D Bin Packing Problem.
Uses continuous position representation mapped to item permutations via argsort,
First-Fit decoding, personal best / global best tracking, velocity clamping,
and strict evaluation budget enforcement.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .validation import validate_packing_solution


def decode_position(
    position: np.ndarray,
    weights: Union[List[int], np.ndarray],
    capacity: int
) -> Tuple[List[List[int]], List[int]]:
    """
    Decode a continuous particle position vector into bins via argsort and First-Fit.

    Parameters
    ----------
    position : np.ndarray
        Continuous vector in [0, 1]^n.
    weights : list or np.ndarray
        Item weights.
    capacity : int
        Bin capacity.

    Returns
    -------
    (list of list of int, list of int)
        Packed bins (indices) and the corresponding bin loads.
    """
    order = np.argsort(position)
    bins: List[List[int]] = []
    bin_loads: List[int] = []

    for item_idx in order:
        w = int(weights[item_idx])
        placed = False
        for b_idx in range(len(bins)):
            if bin_loads[b_idx] + w <= capacity:
                bins[b_idx].append(int(item_idx))
                bin_loads[b_idx] += w
                placed = True
                break

        if not placed:
            bins.append([int(item_idx)])
            bin_loads.append(w)

    return bins, bin_loads


def evaluate_position(
    position: np.ndarray,
    weights: Union[List[int], np.ndarray],
    capacity: int
) -> int:
    """
    Evaluate particle position fitness (number of bins). Minimized.
    """
    bins, _ = decode_position(position, weights, capacity)
    return len(bins)


def run_particle_swarm_optimization(
    weights: Union[List[int], np.ndarray],
    capacity: int,
    population_size: int = 50,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    v_min: float = -0.2,
    v_max: float = 0.2,
    pos_min: float = 0.0,
    pos_max: float = 1.0,
    evaluation_budget: int = 6000,
    seed: Optional[int] = None,
    bks: Optional[int] = None,
) -> Dict:
    """
    Execute custom Particle Swarm Optimization for 1D-BPP with strict evaluation budget.

    Parameters
    ----------
    weights : list of int or np.ndarray
        Item weights.
    capacity : int
        Bin capacity.
    population_size : int
        Swarm size (number of particles).
    w : float
        Inertia weight.
    c1 : float
        Cognitive acceleration coefficient.
    c2 : float
        Social acceleration coefficient.
    v_min, v_max : float
        Velocity clamping bounds.
    pos_min, pos_max : float
        Position boundary clipping bounds.
    evaluation_budget : int
        Maximum objective function evaluations allowed.
    seed : int, optional
        Random seed for deterministic reproducibility.
    bks : int, optional
        Best-Known Solution for computing gap percentage.

    Returns
    -------
    dict
        Execution results including best position, best bins, best fitness,
        evaluations used, full convergence history, runtime, and validity.
    """
    if population_size < 2:
        raise ValueError("Population size must be at least 2.")
    if evaluation_budget < population_size:
        raise ValueError(
            f"Evaluation budget ({evaluation_budget}) must be >= population size ({population_size})."
        )

    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_items = len(weights)
    weights_arr = np.asarray(weights, dtype=int)

    # 1. Initialize swarm positions and velocities
    X = rng.random((population_size, n_items)) * (pos_max - pos_min) + pos_min
    V = rng.uniform(-0.1, 0.1, (population_size, n_items))

    # Evaluate initial swarm
    pbest = X.copy()
    pscore = np.array([
        evaluate_position(x, weights_arr, capacity) for x in X
    ])
    evals = population_size

    # Global best initialization
    best_idx = np.argmin(pscore)
    gbest = pbest[best_idx].copy()
    gscore = int(pscore[best_idx])

    # History: record best-so-far at every single evaluation
    history = [gscore] * population_size

    # 2. Main swarm iteration loop
    while evals < evaluation_budget:
        for i in range(population_size):
            if evals >= evaluation_budget:
                break

            r1 = rng.random(n_items)
            r2 = rng.random(n_items)

            # Velocity update
            V[i] = (
                w * V[i]
                + c1 * r1 * (pbest[i] - X[i])
                + c2 * r2 * (gbest - X[i])
            )

            # Velocity clamping and position bounding
            V[i] = np.clip(V[i], v_min, v_max)
            X[i] = np.clip(X[i] + V[i], pos_min, pos_max)

            # Objective function evaluation
            score = evaluate_position(X[i], weights_arr, capacity)
            evals += 1

            # Personal best update
            if score < pscore[i]:
                pbest[i] = X[i].copy()
                pscore[i] = score

            # Global best update
            if score < gscore:
                gbest = X[i].copy()
                gscore = int(score)

            history.append(gscore)

    runtime = time.perf_counter() - start_time

    # Decode global best solution for full validation
    best_bins, bin_loads = decode_position(gbest, weights_arr, capacity)
    valid, msg = validate_packing_solution(
        bins=best_bins,
        weights=weights_arr,
        capacity=capacity,
        reported_bins=gscore
    )

    gap_percent = None
    if bks is not None and bks > 0:
        gap_percent = (gscore - bks) / bks * 100.0

    return {
        "best_position": gbest,
        "best_bins": best_bins,
        "bin_loads": bin_loads,
        "best_fitness": gscore,
        "evaluations": evals,
        "history": history,
        "runtime_seconds": runtime,
        "valid": valid,
        "validation_message": msg,
        "seed": seed,
        "gap_percent": gap_percent,
        "w": w,
        "c1": c1,
        "c2": c2,
        "population_size": population_size,
    }
