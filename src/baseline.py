"""
First Fit Decreasing (FFD) baseline algorithm for the 1D Bin Packing Problem.
Deterministic, reproducible, and polynomial-time heuristic.
"""

import time
from typing import Dict, List, Optional, Union
import numpy as np

from .validation import validate_packing_solution


def first_fit_decreasing(
    weights: Union[List[int], np.ndarray],
    capacity: int,
    bks: Optional[int] = None
) -> Dict:
    """
    Solve the 1D Bin Packing Problem using First Fit Decreasing (FFD).

    Algorithm:
    1. Sort items in non-increasing order of weights, preserving original indices.
    2. Sequentially place each item into the first bin with sufficient residual capacity.
    3. If no existing bin can accommodate the item, open a new bin.

    Parameters
    ----------
    weights : list of int or np.ndarray
        List of item weights.
    capacity : int
        Capacity of each bin.
    bks : int, optional
        Best-Known Solution / optimum value for computing the gap percentage.

    Returns
    -------
    dict
        - 'bins': list of list of item indices
        - 'bin_loads': list of int, load of each bin
        - 'best_bins': int, total number of bins used
        - 'runtime_seconds': float, execution time
        - 'valid': bool, solution validity
        - 'validation_message': str, validator response
        - 'evaluations': int, 1 (deterministic single heuristic pass)
        - 'gap_percent': float or None, percentage gap to BKS
    """
    start_time = time.perf_counter()

    # Index items [0..n-1] and sort by weight descending (stable sort)
    indexed_items = sorted(
        enumerate(weights),
        key=lambda item: int(item[1]),
        reverse=True
    )

    bins: List[List[int]] = []
    bin_loads: List[int] = []

    for item_idx, weight in indexed_items:
        w = int(weight)
        placed = False
        for b_idx in range(len(bins)):
            if bin_loads[b_idx] + w <= capacity:
                bins[b_idx].append(item_idx)
                bin_loads[b_idx] += w
                placed = True
                break

        if not placed:
            bins.append([item_idx])
            bin_loads.append(w)

    runtime = time.perf_counter() - start_time
    num_bins = len(bins)

    valid, msg = validate_packing_solution(
        bins=bins,
        weights=weights,
        capacity=capacity,
        reported_bins=num_bins
    )

    gap_percent = None
    if bks is not None and bks > 0:
        gap_percent = (num_bins - bks) / bks * 100.0

    return {
        "bins": bins,
        "bin_loads": bin_loads,
        "best_bins": num_bins,
        "runtime_seconds": runtime,
        "valid": valid,
        "validation_message": msg,
        "evaluations": 1,
        "gap_percent": gap_percent,
    }


def first_fit_stepped(
    weights: Union[List[int], np.ndarray],
    capacity: int,
    strategy: str = "decreasing",
    seed: Optional[int] = 42,
    bks: Optional[int] = None
) -> Dict:
    """
    Step-by-step interactive First Fit solver for visual demonstrations.
    Records every decision point (tested bins, placement, or new bin creation).

    Strategies:
    - 'decreasing': First Fit Decreasing (FFD)
    - 'increasing': First Fit Increasing (FFI)
    - 'original': First Fit in original input order
    - 'random': First Fit in random permutation order
    """
    start_time = time.perf_counter()
    n = len(weights)
    indexed_items = list(enumerate(weights))

    if strategy == "decreasing":
        ordered_items = sorted(indexed_items, key=lambda x: int(x[1]), reverse=True)
        strategy_name = "First Fit Decreasing (FFD)"
    elif strategy == "increasing":
        ordered_items = sorted(indexed_items, key=lambda x: int(x[1]), reverse=False)
        strategy_name = "First Fit Increasing (FFI)"
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        ordered_items = [indexed_items[i] for i in perm]
        strategy_name = "First Fit Random"
    else:
        ordered_items = list(indexed_items)
        strategy_name = "Pure First Fit (Original Order)"

    bins: List[List[int]] = []
    bin_loads: List[int] = []

    steps = [
        {
            "step": 0,
            "action_title": "Inisialisasi Algoritma",
            "item_idx": None,
            "weight": None,
            "placed_in_bin": None,
            "is_new_bin": False,
            "trials": [],
            "bins_snapshot": [],
            "bin_loads_snapshot": [],
            "description": f"Strategi: {strategy_name}. Menyiapkan {n} item untuk dimasukkan ke dalam bin (Kapasitas C = {capacity})."
        }
    ]

    for step_num, (item_idx, weight) in enumerate(ordered_items, start=1):
        w = int(weight)
        trials = []
        placed = False
        target_bin = -1
        is_new = False

        for b_idx in range(len(bins)):
            current_load = bin_loads[b_idx]
            new_load = current_load + w
            fits = (new_load <= capacity)
            trials.append({
                "bin_idx": b_idx,
                "current_load": current_load,
                "candidate_load": new_load,
                "fits": fits
            })
            if fits:
                bins[b_idx].append(item_idx)
                bin_loads[b_idx] += w
                placed = True
                target_bin = b_idx
                break

        if not placed:
            bins.append([item_idx])
            bin_loads.append(w)
            target_bin = len(bins) - 1
            is_new = True

        # Generate readable decision explanation
        if is_new:
            if len(bins) == 1:
                desc = f"Item #{item_idx} (Bobot: {w}) dimasukkan ke Bin 1 (Bin pertama dibuka)."
            else:
                desc = (
                    f"Item #{item_idx} (Bobot: {w}) tidak muat di seluruh {len(bins)-1} bin yang ada. "
                    f"Membuka **Bin {target_bin + 1}** baru (Beban saat ini: {w}/{capacity})."
                )
        else:
            checks_count = len(trials)
            if checks_count == 1:
                desc = f"Item #{item_idx} (Bobot: {w}) langsung muat di **Bin {target_bin + 1}** (Beban baru: {bin_loads[target_bin]}/{capacity})."
            else:
                desc = (
                    f"Item #{item_idx} (Bobot: {w}) diperiksa ke {checks_count - 1} bin sebelumnya (tidak cukup ruang), "
                    f"lalu berhasil dimasukkan ke **Bin {target_bin + 1}** (Beban baru: {bin_loads[target_bin]}/{capacity})."
                )

        steps.append({
            "step": step_num,
            "action_title": f"Langkah {step_num}/{n}: Item #{item_idx} (Bobot: {w})",
            "item_idx": item_idx,
            "weight": w,
            "placed_in_bin": target_bin,
            "is_new_bin": is_new,
            "trials": trials,
            "bins_snapshot": [b.copy() for b in bins],
            "bin_loads_snapshot": list(bin_loads),
            "description": desc
        })

    runtime = time.perf_counter() - start_time
    num_bins = len(bins)

    valid, msg = validate_packing_solution(
        bins=bins,
        weights=weights,
        capacity=capacity,
        reported_bins=num_bins
    )

    gap_percent = None
    if bks is not None and bks > 0:
        gap_percent = (num_bins - bks) / bks * 100.0

    return {
        "strategy": strategy,
        "strategy_name": strategy_name,
        "ordered_items": ordered_items,
        "steps": steps,
        "bins": bins,
        "bin_loads": bin_loads,
        "best_bins": num_bins,
        "runtime_seconds": runtime,
        "valid": valid,
        "validation_message": msg,
        "evaluations": 1,
        "gap_percent": gap_percent,
    }

