"""
Unit and integration tests for Particle Swarm Optimization (PSO) module.
"""

from pathlib import Path
import numpy as np
from src.data_loader import load_instance
from src.pso import (
    decode_position,
    run_particle_swarm_optimization,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Falkenauer_U"


def test_pso_decoding_preserves_items_and_capacity():
    weights = [30, 40, 50, 20, 60]
    capacity = 100
    position = np.array([0.9, 0.1, 0.4, 0.8, 0.2])

    bins, bin_loads = decode_position(position, weights, capacity)
    all_indices = [idx for b in bins for idx in b]

    assert sorted(all_indices) == list(range(len(weights)))
    assert all(load <= capacity for load in bin_loads)


def test_pso_budget_enforcement():
    inst = load_instance(DATA_DIR / "Falkenauer_u120_00.txt")
    budget = 400
    pop_size = 50

    res = run_particle_swarm_optimization(
        weights=inst["weights"],
        capacity=inst["capacity"],
        population_size=pop_size,
        evaluation_budget=budget,
        seed=999
    )

    assert res["evaluations"] == budget
    assert len(res["history"]) == budget
    assert res["valid"] is True


def test_pso_reproducibility_deterministic_seed():
    inst = load_instance(DATA_DIR / "Falkenauer_u120_00.txt")

    res1 = run_particle_swarm_optimization(
        weights=inst["weights"],
        capacity=inst["capacity"],
        population_size=50,
        evaluation_budget=300,
        seed=77
    )

    res2 = run_particle_swarm_optimization(
        weights=inst["weights"],
        capacity=inst["capacity"],
        population_size=50,
        evaluation_budget=300,
        seed=77
    )

    assert res1["best_fitness"] == res2["best_fitness"]
    np.testing.assert_array_equal(res1["best_position"], res2["best_position"])
    assert res1["history"] == res2["history"]
    assert res1["evaluations"] == res2["evaluations"]
