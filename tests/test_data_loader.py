"""
Unit tests for data_loader module.
"""

from pathlib import Path
import pytest

from src.data_loader import load_instance, load_all_instances

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Falkenauer_U"


def test_load_single_instance():
    inst_path = DATA_DIR / "Falkenauer_u120_00.txt"
    inst = load_instance(inst_path)

    assert inst["instance_name"] == "Falkenauer_u120_00.txt"
    assert inst["n_items"] == 120
    assert inst["capacity"] == 150
    assert len(inst["weights"]) == 120
    assert inst["total_weight"] == 7078
    assert inst["volume_lower_bound"] == 48
    assert all(w > 0 for w in inst["weights"])
    assert all(w <= 150 for w in inst["weights"])


def test_load_all_five_instances():
    expected_bks = {
        "Falkenauer_u120_00.txt": 48,
        "Falkenauer_u120_01.txt": 49,
        "Falkenauer_u120_02.txt": 46,
        "Falkenauer_u120_03.txt": 49,
        "Falkenauer_u120_04.txt": 50,
    }
    instances = load_all_instances(DATA_DIR, list(expected_bks.keys()))
    assert len(instances) == 5

    for name, bks in expected_bks.items():
        assert name in instances
        assert instances[name]["n_items"] == 120
        assert instances[name]["capacity"] == 150
        assert instances[name]["volume_lower_bound"] <= bks


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_instance(DATA_DIR / "non_existent_file.txt")
