"""
Shared solution validation module for Bin Packing Problem solutions.
Ensures rigorous verification using item indices to handle items with identical weights.
"""

from typing import List, Tuple, Union
import numpy as np


def validate_packing_solution(
    bins: List[List[int]],
    weights: Union[List[int], np.ndarray],
    capacity: int,
    reported_bins: int = None
) -> Tuple[bool, str]:
    """
    Validate a 1D Bin Packing Problem solution.

    A valid solution must satisfy four conditions:
    1. Every item index (0 to n - 1) appears in the packing.
    2. No item index is duplicated (each appears exactly once).
    3. The sum of item weights in every bin does not exceed bin capacity.
    4. The number of bins in the packing matches the reported objective value.

    Parameters
    ----------
    bins : list of list of int
        List of bins, where each bin is a list of original item indices [0..n-1].
    weights : list of int or np.ndarray
        List or array of original item weights.
    capacity : int
        Maximum capacity of a bin.
    reported_bins : int, optional
        The reported objective value (number of bins) to verify consistency.

    Returns
    -------
    (bool, str)
        Tuple of (is_valid, message).
    """
    n_items = len(weights)

    if reported_bins is not None and len(bins) != reported_bins:
        return False, (
            f"Objective mismatch: packed bins count ({len(bins)}) != "
            f"reported bins ({reported_bins})."
        )

    # 1 & 2. Item conservation check using indices
    all_indices = [idx for b in bins for idx in b]

    if len(all_indices) != n_items:
        return False, (
            f"Item count mismatch: total packed items ({len(all_indices)}) != "
            f"expected items ({n_items})."
        )

    # Check for duplicate indices
    seen = set()
    duplicates = []
    for idx in all_indices:
        if not isinstance(idx, (int, np.integer)):
            return False, f"Item index must be an integer, got {type(idx)} ({idx})."
        if idx < 0 or idx >= n_items:
            return False, f"Item index {idx} out of range [0, {n_items - 1}]."
        if idx in seen:
            duplicates.append(idx)
        seen.add(idx)

    if duplicates:
        return False, f"Duplicate item indices found: {duplicates[:5]}."

    if len(seen) != n_items:
        missing = set(range(n_items)) - seen
        return False, f"Missing item indices: {list(missing)[:5]}."

    # 3. Capacity feasibility check
    for b_idx, b in enumerate(bins):
        if len(b) == 0:
            return False, f"Bin {b_idx + 1} is empty."
        bin_load = sum(int(weights[idx]) for idx in b)
        if bin_load > capacity:
            return False, (
                f"Bin {b_idx + 1} exceeds capacity: load={bin_load} > capacity={capacity}."
            )

    return True, "Solution is valid: all items packed exactly once and capacity respected."
