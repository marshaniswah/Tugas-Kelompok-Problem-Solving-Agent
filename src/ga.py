"""
Custom Genetic Algorithm (GA) for the 1D Bin Packing Problem.
Uses permutation representation, First-Fit decoding, tournament selection,
Order Crossover (OX), swap mutation, and elitism.
Deterministic and strictly budget-enforced.
"""

import random
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .validation import validate_packing_solution


def decode_permutation(
    chromosome: List[int],
    weights: Union[List[int], np.ndarray],
    capacity: int
) -> Tuple[List[List[int]], List[int]]:
    """
    Decode an item permutation chromosome into bins using First-Fit heuristic.

    Parameters
    ----------
    chromosome : list of int
        Permutation of item indices [0..n-1].
    weights : list of int or np.ndarray
        Weights of items.
    capacity : int
        Bin capacity.

    Returns
    -------
    (list of list of int, list of int)
        Packed bins (indices) and the corresponding bin loads.
    """
    bins: List[List[int]] = []
    bin_loads: List[int] = []

    for item_idx in chromosome:
        w = int(weights[item_idx])
        placed = False
        for b in range(len(bins)):
            if bin_loads[b] + w <= capacity:
                bins[b].append(item_idx)
                bin_loads[b] += w
                placed = True
                break

        if not placed:
            bins.append([item_idx])
            bin_loads.append(w)

    return bins, bin_loads


def evaluate_chromosome(
    chromosome: List[int],
    weights: Union[List[int], np.ndarray],
    capacity: int
) -> int:
    """
    Evaluate chromosome fitness (number of bins used). Minimized.
    """
    bins, _ = decode_permutation(chromosome, weights, capacity)
    return len(bins)


def initialize_population(
    n_items: int,
    pop_size: int,
    rng: random.Random
) -> List[List[int]]:
    """
    Initialize a population of random item permutations.
    """
    base = list(range(n_items))
    population = []
    for _ in range(pop_size):
        chrom = base.copy()
        rng.shuffle(chrom)
        population.append(chrom)
    return population


def tournament_selection(
    population: List[List[int]],
    fitness_values: List[int],
    tournament_size: int,
    rng: random.Random
) -> List[int]:
    """
    Select an individual using tournament selection (minimization).
    """
    candidate_indices = rng.sample(range(len(population)), tournament_size)
    winner_idx = min(candidate_indices, key=lambda i: fitness_values[i])
    return population[winner_idx].copy()


def order_crossover(
    parent1: List[int],
    parent2: List[int],
    cut1: int,
    cut2: int
) -> List[int]:
    """
    Order Crossover (OX) for permutation chromosomes.

    Preserves the relative ordering and ensures each item index appears exactly once.
    Copies parent1[cut1 : cut2 + 1] into child, then fills remaining slots circularly from parent2.
    """
    n = len(parent1)
    if not (0 <= cut1 <= cut2 < n):
        raise ValueError(f"Invalid cut points: cut1={cut1}, cut2={cut2}, n={n}")

    child = [None] * n
    # 1. Copy segment from parent1
    child[cut1: cut2 + 1] = parent1[cut1: cut2 + 1]

    # 2. Circular order of genes from parent2 starting after cut2
    p2_order = parent2[cut2 + 1:] + parent2[:cut2 + 1]

    # 3. Filter out genes already in child segment
    copied_set = set(parent1[cut1: cut2 + 1])
    remaining = [gene for gene in p2_order if gene not in copied_set]

    # 4. Fill positions circularly after cut2
    positions = list(range(cut2 + 1, n)) + list(range(0, cut1))
    for pos, gene in zip(positions, remaining):
        child[pos] = gene

    return child


def swap_mutation(
    chromosome: List[int],
    mutation_rate: float,
    rng: random.Random
) -> List[int]:
    """
    Swap mutation: swaps two random genes with probability `mutation_rate`.
    Always maintains a valid permutation.
    """
    mutated = chromosome.copy()
    if rng.random() < mutation_rate and len(mutated) >= 2:
        i, j = rng.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def run_genetic_algorithm(
    weights: Union[List[int], np.ndarray],
    capacity: int,
    population_size: int = 50,
    tournament_size: int = 3,
    mutation_rate: float = 0.20,
    crossover_rate: float = 0.90,
    evaluation_budget: int = 6000,
    seed: Optional[int] = None,
    bks: Optional[int] = None,
) -> Dict:
    """
    Execute Genetic Algorithm for 1D-BPP with strict objective evaluation budget.

    Parameters
    ----------
    weights : list of int or np.ndarray
        Item weights.
    capacity : int
        Bin capacity.
    population_size : int
        Number of chromosomes per generation.
    tournament_size : int
        Number of competitors in tournament selection.
    mutation_rate : float
        Probability of applying swap mutation per child.
    crossover_rate : float
        Probability of applying Order Crossover (OX).
    evaluation_budget : int
        Maximum number of objective function evaluations allowed.
    seed : int, optional
        Random seed for deterministic reproducibility.
    bks : int, optional
        Best-Known Solution for computing gap percentage.

    Returns
    -------
    dict
        Execution results including best chromosome, best bins, best fitness,
        evaluations used, convergence history, checkpoints, runtime, and validity.
    """
    if population_size < 2:
        raise ValueError("Population size must be at least 2.")
    if evaluation_budget < population_size:
        raise ValueError(
            f"Evaluation budget ({evaluation_budget}) must be >= population size ({population_size})."
        )

    start_time = time.perf_counter()
    rng = random.Random(seed)
    n_items = len(weights)

    # 1. Initialize and evaluate initial population
    population = initialize_population(
        n_items=n_items,
        pop_size=population_size,
        rng=random.Random(rng.randrange(0, 2**32))
    )

    fitness_values = [
        evaluate_chromosome(chrom, weights, capacity)
        for chrom in population
    ]
    evaluations = population_size

    # Best-so-far tracking
    best_idx = min(range(population_size), key=lambda i: fitness_values[i])
    best_chromosome = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    convergence = [best_fitness]
    eval_checkpoints = [evaluations]
    mean_history = [float(np.mean(fitness_values))]

    # 2. Main generational loop
    while evaluations + population_size <= evaluation_budget:
        # Elitism: retain single best individual
        next_population = [best_chromosome.copy()]

        while len(next_population) < population_size:
            parent1 = tournament_selection(
                population, fitness_values, tournament_size,
                rng=random.Random(rng.randrange(0, 2**32))
            )
            parent2 = tournament_selection(
                population, fitness_values, tournament_size,
                rng=random.Random(rng.randrange(0, 2**32))
            )

            if rng.random() < crossover_rate:
                cut1, cut2 = sorted(rng.sample(range(n_items), 2))
                child = order_crossover(parent1, parent2, cut1, cut2)
            else:
                child = parent1.copy()

            child = swap_mutation(
                child, mutation_rate,
                rng=random.Random(rng.randrange(0, 2**32))
            )
            next_population.append(child)

        population = next_population

        # Evaluate offspring
        fitness_values = [
            evaluate_chromosome(chrom, weights, capacity)
            for chrom in population
        ]
        evaluations += population_size

        curr_best_idx = min(range(population_size), key=lambda i: fitness_values[i])
        curr_best = fitness_values[curr_best_idx]

        if curr_best < best_fitness:
            best_fitness = curr_best
            best_chromosome = population[curr_best_idx].copy()

        convergence.append(best_fitness)
        eval_checkpoints.append(evaluations)
        mean_history.append(float(np.mean(fitness_values)))

    runtime = time.perf_counter() - start_time

    # Decode best solution for full validation
    best_bins, bin_loads = decode_permutation(best_chromosome, weights, capacity)
    valid, msg = validate_packing_solution(
        bins=best_bins,
        weights=weights,
        capacity=capacity,
        reported_bins=best_fitness
    )

    gap_percent = None
    if bks is not None and bks > 0:
        gap_percent = (best_fitness - bks) / bks * 100.0

    return {
        "best_chromosome": best_chromosome,
        "best_bins": best_bins,
        "bin_loads": bin_loads,
        "best_fitness": best_fitness,
        "evaluations": evaluations,
        "generations_completed": len(convergence) - 1,
        "convergence": convergence,
        "eval_checkpoints": eval_checkpoints,
        "mean_fitness_history": mean_history,
        "runtime_seconds": runtime,
        "valid": valid,
        "validation_message": msg,
        "seed": seed,
        "gap_percent": gap_percent,
        "population_size": population_size,
        "tournament_size": tournament_size,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
    }


def run_genetic_algorithm_stepped(
    weights: Union[List[int], np.ndarray],
    capacity: int,
    population_size: int = 30,
    generations: int = 30,
    tournament_size: int = 3,
    mutation_rate: float = 0.20,
    crossover_rate: float = 0.90,
    elitism: bool = True,
    seed: Optional[int] = 42,
    bks: Optional[int] = None
) -> Dict:
    """
    Generation-by-generation Genetic Algorithm solver for interactive visualization.
    Records snapshots of each generation including population fitness distribution,
    best chromosome, and decoded packing layout.
    """
    if population_size < 2:
        raise ValueError("Population size must be at least 2.")
    if tournament_size < 1 or tournament_size > population_size:
        tournament_size = min(max(1, tournament_size), population_size)

    start_time = time.perf_counter()
    rng = random.Random(seed)
    n_items = len(weights)

    # 1. Initialize Generation 0
    population = initialize_population(
        n_items=n_items,
        pop_size=population_size,
        rng=random.Random(rng.randrange(0, 2**32))
    )

    fitness_values = [
        evaluate_chromosome(chrom, weights, capacity)
        for chrom in population
    ]
    evaluations = population_size

    best_idx = min(range(population_size), key=lambda i: fitness_values[i])
    best_chromosome = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    best_bins, bin_loads = decode_permutation(best_chromosome, weights, capacity)

    generations_data = [
        {
            "generation": 0,
            "evaluations": evaluations,
            "best_fitness": best_fitness,
            "best_chromosome": best_chromosome.copy(),
            "best_bins": [b.copy() for b in best_bins],
            "bin_loads": list(bin_loads),
            "mean_fitness": float(np.mean(fitness_values)),
            "min_fitness": int(np.min(fitness_values)),
            "max_fitness": int(np.max(fitness_values)),
            "fitness_distribution": list(fitness_values),
            "description": (
                f"**Generasi 0 (Inisialisasi Acak)**: Membangkitkan {population_size} kromosom permutasi. "
                f"Solusi acak terbaik awal membutuhkan **{best_fitness} bin** (Rata-rata: {np.mean(fitness_values):.1f} bin)."
            )
        }
    ]

    convergence = [best_fitness]
    mean_history = [float(np.mean(fitness_values))]

    # 2. Generational Evolution Loop
    for gen in range(1, generations + 1):
        prev_best = best_fitness
        next_population = [best_chromosome.copy()] if elitism else []

        while len(next_population) < population_size:
            p1 = tournament_selection(
                population, fitness_values, tournament_size,
                rng=random.Random(rng.randrange(0, 2**32))
            )
            p2 = tournament_selection(
                population, fitness_values, tournament_size,
                rng=random.Random(rng.randrange(0, 2**32))
            )

            if rng.random() < crossover_rate and n_items >= 2:
                cut1, cut2 = sorted(rng.sample(range(n_items), 2))
                child = order_crossover(p1, p2, cut1, cut2)
            else:
                child = p1.copy()

            child = swap_mutation(
                child, mutation_rate,
                rng=random.Random(rng.randrange(0, 2**32))
            )
            next_population.append(child)

        population = next_population
        fitness_values = [
            evaluate_chromosome(chrom, weights, capacity)
            for chrom in population
        ]
        evaluations += len(population)

        curr_best_idx = min(range(population_size), key=lambda i: fitness_values[i])
        curr_best = fitness_values[curr_best_idx]

        improved = False
        if curr_best < best_fitness:
            best_fitness = curr_best
            best_chromosome = population[curr_best_idx].copy()
            best_bins, bin_loads = decode_permutation(best_chromosome, weights, capacity)
            improved = True

        convergence.append(best_fitness)
        mean_history.append(float(np.mean(fitness_values)))

        if improved:
            desc = (
                f"✨ **Peningkatan Ditemukan di Generasi {gen}**! Solusi terbaik berkurang dari {prev_best} menjadi **{best_fitness} bin**. "
                f"Kombinasi crossover & mutasi berhasil mengeliminasi ruang kosong (slack)."
            )
        else:
            desc = (
                f"**Generasi {gen}**: Seleksi turnamen (k={tournament_size}), OX crossover (p_c={crossover_rate:.2f}), "
                f"dan mutasi swap (p_m={mutation_rate:.2f}). Solusi terbaik bertahan pada **{best_fitness} bin** (Rata-rata: {np.mean(fitness_values):.1f} bin)."
            )

        generations_data.append({
            "generation": gen,
            "evaluations": evaluations,
            "best_fitness": best_fitness,
            "best_chromosome": best_chromosome.copy(),
            "best_bins": [b.copy() for b in best_bins],
            "bin_loads": list(bin_loads),
            "mean_fitness": float(np.mean(fitness_values)),
            "min_fitness": int(np.min(fitness_values)),
            "max_fitness": int(np.max(fitness_values)),
            "fitness_distribution": list(fitness_values),
            "description": desc
        })

    runtime = time.perf_counter() - start_time

    valid, msg = validate_packing_solution(
        bins=best_bins,
        weights=weights,
        capacity=capacity,
        reported_bins=best_fitness
    )

    gap_percent = None
    if bks is not None and bks > 0:
        gap_percent = (best_fitness - bks) / bks * 100.0

    return {
        "best_chromosome": best_chromosome,
        "best_bins": best_bins,
        "bin_loads": bin_loads,
        "best_fitness": best_fitness,
        "evaluations": evaluations,
        "generations_completed": generations,
        "generations_data": generations_data,
        "convergence": convergence,
        "mean_fitness_history": mean_history,
        "runtime_seconds": runtime,
        "valid": valid,
        "validation_message": msg,
        "seed": seed,
        "gap_percent": gap_percent,
        "population_size": population_size,
        "tournament_size": tournament_size,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "elitism": elitism
    }

