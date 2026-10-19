"""
Unit tests for baseline First Fit Decreasing (FFD) module.
"""

from pathlib import Path
from src.data_loader import load_instance
from src.baseline import first_fit_decreasing
from src.metrics import compute_gap_percent

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Falkenauer_U"


def test_ffd_toy_example():
    weights = [4, 8, 1, 4, 2, 1]
    capacity = 10
    # Sorted descending: [8, 4, 4, 2, 1, 1]
    # Bin 1: 8 + 2 = 10
    # Bin 2: 4 + 4 + 1 + 1 = 10
    # Total bins = 2
    res = first_fit_decreasing(weights, capacity, bks=2)

    assert res["best_bins"] == 2
    assert res["valid"] is True
    assert res["gap_percent"] == 0.0
    assert len(res["bin_loads"]) == 2
    assert all(load <= capacity for load in res["bin_loads"])


def test_ffd_on_falkenauer_instances():
    expected_results = {
        "Falkenauer_u120_00.txt": {"bks": 48, "bins": 49},
        "Falkenauer_u120_01.txt": {"bks": 49, "bins": 49},
        "Falkenauer_u120_02.txt": {"bks": 46, "bins": 47},
        "Falkenauer_u120_03.txt": {"bks": 49, "bins": 50},
        "Falkenauer_u120_04.txt": {"bks": 50, "bins": 50},
    }

    for fname, exp in expected_results.items():
        inst = load_instance(DATA_DIR / fname)
        res = first_fit_decreasing(inst["weights"], inst["capacity"], bks=exp["bks"])

        assert res["best_bins"] == exp["bins"]
        assert res["valid"] is True
        expected_gap = (exp["bins"] - exp["bks"]) / exp["bks"] * 100.0
        assert round(res["gap_percent"], 2) == round(expected_gap, 2)
