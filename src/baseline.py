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
