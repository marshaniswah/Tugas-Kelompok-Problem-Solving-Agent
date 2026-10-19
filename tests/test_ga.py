"""
Unit and integration tests for Genetic Algorithm (GA) module.
"""

import random
from pathlib import Path
from src.data_loader import load_instance
from src.ga import (
    initialize_population,
    order_crossover,
    swap_mutation,
    run_genetic_algorithm,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Falkenauer_U"


def test_ga_operators_preserve_permutation():
    n_items = 20
    rng = random.Random(42)
    pop = initialize_population(n_items, 10, rng)

    assert len(pop) == 10
    for chrom in pop:
        assert len(chrom) == n_items
        assert sorted(chrom) == list(range(n_items))

    # Test Order Crossover
    p1 = pop[0]
    p2 = pop[1]
    child = order_crossover(p1, p2, cut1=4, cut2=12)

    assert len(child) == n_items
    assert sorted(child) == list(range(n_items))
    assert child[4:13] == p1[4:13]

    # Test Swap Mutation
    mutated = swap_mutation(child, mutation_rate=1.0, rng=rng)
    assert len(mutated) == n_items
    assert sorted(mutated) == list(range(n_items))


def test_ga_budget_enforcement():
    inst = load_instance(DATA_DIR / "Falkenauer_u120_00.txt")
    budget = 500
    pop_size = 50

    res = run_genetic_algorithm(
        weights=inst["weights"],
        capacity=inst["capacity"],
        population_size=pop_size,
        evaluation_budget=budget,
        seed=123
    )

    assert res["evaluations"] <= budget
    assert res["evaluations"] == budget  # 500 is divisible by 50
    assert res["valid"] is True
    assert len(res["convergence"]) == budget // pop_size


def test_ga_reproducibility_deterministic_seed():
    inst = load_instance(DATA_DIR / "Falkenauer_u120_00.txt")

    res1 = run_genetic_algorithm(
        weights=inst["weights"],
        capacity=inst["capacity"],
        population_size=50,
        evaluation_budget=300,
        seed=42
    )

    res2 = run_genetic_algorithm(
        weights=inst["weights"],
        capacity=inst["capacity"],
        population_size=50,
        evaluation_budget=300,
        seed=42
    )

    assert res1["best_fitness"] == res2["best_fitness"]
    assert res1["best_chromosome"] == res2["best_chromosome"]
    assert res1["convergence"] == res2["convergence"]
    assert res1["evaluations"] == res2["evaluations"]
