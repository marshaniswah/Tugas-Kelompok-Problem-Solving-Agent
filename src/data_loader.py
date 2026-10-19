"""
Data loader and validation module for Bin Packing Problem (BPP) benchmark instances.
Supports the Falkenauer U format and BPPLIB standard format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import math


def load_instance(filepath: Union[str, Path]) -> Dict[str, Union[str, int, List[int]]]:
    """
    Load a single 1D-BPP instance file.

    File format:
        Line 1: Number of items (n)
        Line 2: Bin capacity (C)
        Lines 3..n+2: Weights of the items

    Parameters
    ----------
    filepath : str or Path
        Path to the instance .txt file.

    Returns
    -------
    dict
        Dictionary containing:
        - 'instance_name': file name
        - 'filepath': str path
        - 'n_items': int, number of items
        - 'capacity': int, bin capacity
        - 'weights': list of int, item weights
        - 'total_weight': int, sum of all item weights
        - 'volume_lower_bound': int, ceil(total_weight / capacity)

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If file structure or data constraints are violated.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path.resolve()}")

    content = path.read_text(encoding="utf-8").split()
    if len(content) < 3:
        raise ValueError(f"Instance file {path.name} is malformed or empty.")

    try:
        values = [int(token) for token in content]
    except ValueError as e:
        raise ValueError(f"Failed to parse integers in {path.name}: {e}") from e

    n_items = values[0]
    capacity = values[1]
    weights = values[2: 2 + n_items]

    if len(weights) != n_items:
        raise ValueError(
            f"Item count mismatch in {path.name}: expected {n_items}, parsed {len(weights)}."
        )

    if capacity <= 0:
        raise ValueError(f"Bin capacity must be positive, got {capacity} in {path.name}.")

    for idx, w in enumerate(weights):
        if w <= 0:
            raise ValueError(f"Item weight at index {idx} is non-positive ({w}) in {path.name}.")
        if w > capacity:
            raise ValueError(
                f"Item weight at index {idx} ({w}) exceeds capacity ({capacity}) in {path.name}."
            )

    total_weight = sum(weights)
    vol_lb = math.ceil(total_weight / capacity)

    return {
        "instance_name": path.name,
        "filepath": str(path.resolve()),
        "n_items": n_items,
        "capacity": capacity,
        "weights": weights,
        "total_weight": total_weight,
        "volume_lower_bound": vol_lb,
    }


def load_all_instances(
    data_dir: Union[str, Path],
    filenames: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Load multiple BPP instances from a directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing instance files.
    filenames : list of str, optional
        Specific filenames to load. If None, loads all .txt files.

    Returns
    -------
    dict
        Mapping of filename to instance dictionary.
    """
    dir_path = Path(data_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {dir_path.resolve()}")

    if filenames is None:
        target_files = sorted(dir_path.glob("*.txt"))
    else:
        target_files = [dir_path / fname for fname in filenames]

    instances = {}
    for f in target_files:
        inst = load_instance(f)
        instances[inst["instance_name"]] = inst

    return instances
