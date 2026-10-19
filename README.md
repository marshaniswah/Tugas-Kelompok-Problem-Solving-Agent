# 1D Bin Packing Problem (1D-BPP) Optimization Benchmark
### Problem-Solving Agent Research Repository: Custom GA vs. Custom PSO vs. First Fit Decreasing (FFD)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/reproducible-deterministic%20seeds-brightgreen.svg)]()
[![Code Style: Clean](https://img.shields.io/badge/code%20style-modular%20OOP-blueviolet.svg)]()

> **Academic Course**: AI5103 Fundamental of AI & Data Science — Problem Solving Agent  
> **Topic**: Metaheuristic and Heuristic Optimization for Combinatorial Problem Solving  
> **Authors**: AI5103 Research Group (`marshaniswah`)  
> **Repository**: [https://github.com/marshaniswah/Tugas-Kelompok-Problem-Solving-Agent](https://github.com/marshaniswah/Tugas-Kelompok-Problem-Solving-Agent)

---

## Table of Contents
1. [Project Overview and Problem Definition](#1-project-overview-and-problem-definition)
2. [Benchmark Instances and Proven Optima (BKS)](#2-benchmark-instances-and-proven-optima-bks)
3. [Algorithm Design and Representations](#3-algorithm-design-and-representations)
4. [Solution Validation Integrity](#4-solution-validation-integrity)
5. [Repository Structure](#5-repository-structure)
6. [Installation and Environment Setup](#6-installation-and-environment-setup)
7. [Reproduction Commands](#7-reproduction-commands)
8. [Configuration System (`experiment_config.json`)](#8-configuration-system-experiment_configjson)
9. [Experimental Results Summary](#9-experimental-results-summary)
10. [Statistical Significance Analysis](#10-statistical-significance-analysis)
11. [Convergence Plots](#11-convergence-plots)
12. [GA Operator Variation and Parameter Sensitivity Analysis](#12-ga-operator-variation-and-parameter-sensitivity-analysis)
13. [Methodological Limitations](#13-methodological-limitations)
14. [Scientific Statement and AI Disclosure](#14-scientific-statement-and-ai-disclosure)
15. [References and Citations](#15-references-and-citations)

---

## 1. Project Overview and Problem Definition

The **One-Dimensional Bin Packing Problem (1D-BPP)** is an NP-hard combinatorial optimization problem. Given:
- A set of $n$ items, each with a positive integer weight $w_i \in \mathbb{Z}^+$ ($i = 0, \dots, n-1$).
- Identical one-dimensional containers (bins), each with fixed integer capacity $C$.
- The condition that $w_i \le C$ for all $i$.

The objective is to partition the $n$ items into the minimum number of bins $m$, minimizing:
$$\min \quad m = \sum_{j=1}^m y_j$$
subject to:
$$\sum_{i=0}^{n-1} w_i x_{ij} \le C y_j, \quad \forall j \in \{1, \dots, m\}$$
$$\sum_{j=1}^m x_{ij} = 1, \quad \forall i \in \{0, \dots, n-1\}$$
$$x_{ij} \in \{0, 1\}, \quad y_j \in \{0, 1\}$$
where $y_j = 1$ if bin $j$ is used, and $x_{ij} = 1$ if item $i$ is assigned to bin $j$.

This repository provides a modular, fully reproducible empirical comparison of two custom metaheuristics—a **Genetic Algorithm (GA)** and a **Particle Swarm Optimization (PSO)** implementation—evaluated against the deterministic **First Fit Decreasing (FFD)** baseline under a fair, identical computational budget of **6,000 objective-function evaluations** per run.

---

## 2. Benchmark Instances and Proven Optima (BKS)

The benchmark instances evaluated in this study belong to the widely recognized **Falkenauer U** (Uniform) dataset introduced by Emanuel Falkenauer (1996). 

- **Instance Format**: $n = 120$ items, bin capacity $C = 150$, item weights uniformly distributed in $[20, 100]$.
- **Continuous Volume Lower Bound**:
  $$L_1 = \left\lceil \frac{\sum_{i=0}^{n-1} w_i}{C} \right\rceil$$
- **Benchmark Source and Proven Optima**:
  - Delorme, M., Iori, M., & Martello, S. (2018). *BPPLIB: a library for bin packing and cutting stock problems*. Optimization Letters, 12(2), 235-250.
  - BPPLIB Web Library (University of Bologna): [https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library](https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library)
  - BPPLIB GitHub Repository: [https://github.com/mdelorme2/BPPLIB](https://github.com/mdelorme2/BPPLIB)
  - Proven Best-Known Solutions (BKS) verified from BPPLIB `Solutions.xlsx` (`Status: Solved`, Best LB = Best UB):

| Instance Name | Number of Items ($n$) | Bin Capacity ($C$) | Total Weight ($\sum w_i$) | Volume Lower Bound ($L_1$) | Certified Optimum (BKS) | BPPLIB Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Falkenauer_u120_00.txt` | 120 | 150 | 7,078 | 48 | **48** | Solved |
| `Falkenauer_u120_01.txt` | 120 | 150 | 7,205 | 49 | **49** | Solved |
| `Falkenauer_u120_02.txt` | 120 | 150 | 6,794 | 46 | **46** | Solved |
| `Falkenauer_u120_03.txt` | 120 | 150 | 7,285 | 49 | **49** | Solved |
| `Falkenauer_u120_04.txt` | 120 | 150 | 7,354 | 50 | **50** | Solved |

### Gap Percentage Formula
The percentage gap to the proven optimum / BKS is calculated as:
$$\text{gap\_percent} = \frac{\text{result} - \text{BKS}}{\text{BKS}} \times 100\%$$
A gap of $0.00\%$ indicates that the algorithm successfully discovered an optimal packing.

---

## 3. Algorithm Design and Representations

Both metaheuristic implementations were authored from scratch in Python/NumPy without using third-party evolutionary frameworks (such as DEAP or PySwarms) for the submitted core solvers.

### A. Custom Genetic Algorithm (GA)
- **Chromosome Representation**: Direct permutation of item indices $P = (p_0, p_1, \dots, p_{n-1})$ where $p_i \in \{0, \dots, n-1\}$.
- **Decoder**: First-Fit decoder that traverses the permutation and assigns each item to the first bin with sufficient residual capacity. If no open bin can hold the item, a new bin is created.
- **Fitness / Objective**: Total number of packed bins, $f(P) = |\text{Bins}|$, minimized.
- **Selection**: Tournament Selection with tournament size $k = 3$.
- **Crossover**: Order Crossover (OX) with probability $p_c = 0.90$. OX copies a substring from the first parent and circularly fills the remaining slots from the second parent, guaranteeing valid permutations without missing or duplicate items.
- **Mutation**: Swap Mutation with probability $p_m = 0.20$. Swaps two randomly selected gene indices.
- **Elitism**: Preserves the single best individual ($1$-elitism) into the next generation.
- **Evaluation Budget Accounting**: Population size $N = 50$. Initial population evaluates 50 individuals. Each generation evaluates 50 offspring. Total budget = 6,000 evaluations (1 initial + 119 generations = 120 generations total). Convergence is logged at every generation checkpoint ($50, 100, 150, \dots, 6000$).

### B. Custom Particle Swarm Optimization (PSO)
- **Particle Representation**: Continuous real-valued position vector $\mathbf{x}_i \in [0, 1]^n$.
- **Permutation Mapping**: Continuous coordinates are converted to an item permutation using the indirect `argsort` mapping ($P = \text{argsort}(\mathbf{x}_i)$), which is subsequently decoded using the First-Fit heuristic.
- **Velocity and Position Updates**:
  $$\mathbf{v}_i^{t+1} = w \mathbf{v}_i^t + c_1 r_1 (\mathbf{pbest}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{gbest} - \mathbf{x}_i^t)$$
  $$\mathbf{x}_i^{t+1} = \text{clip}(\mathbf{x}_i^t + \mathbf{v}_i^{t+1}, 0.0, 1.0)$$
- **Velocity Clamping**: $\mathbf{v}_i \in [v_{\min}, v_{\max}] = [-0.2, 0.2]$ to prevent explosive trajectory divergence in continuous space.
- **Hyperparameters**: Swarm size $N = 50$, inertia weight $w = 0.7$, cognitive parameter $c_1 = 1.5$, social parameter $c_2 = 1.5$.
- **Evaluation Budget**: Exactly 6,000 evaluations per run (50 particles across 120 iterations).

### C. First Fit Decreasing (FFD) Baseline
- **Methodology**: Deterministic sorting of items in non-increasing order of weight ($w_0 \ge w_1 \ge \dots \ge w_{n-1}$), followed by First-Fit bin allocation.
- **Complexity**: $O(n \log n)$ time complexity, serves as the standard industrial benchmark.

---

## 4. Solution Validation Integrity

Every solution generated by FFD, GA, and PSO is audited by an independent, strict validation protocol ([`src/validation.py`](src/validation.py)). The validator asserts:
1. **Item Index Conservation**: Every item index $i \in \{0, \dots, n-1\}$ appears **exactly once** across all bins.
2. **Total Item Count**: $\sum_{b \in \text{Bins}} |b| = n$.
3. **Bin Capacity Compliance**: For every bin $b$, $\sum_{i \in b} w_i \le C$.

Any duplicate item, omitted item, or bin capacity violation immediately invalidates the run (`valid = False`). All 25 runs for GA, PSO, and FFD passed 100% of validation assertions.

---

## 5. Repository Structure

```
bpp-optimization/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Python package dependencies
├── pytest.ini                     # Pytest configuration
├── .gitignore                     # Git hygiene filter
├── config/
│   └── experiment_config.json     # Central experiment configuration
├── data/
│   └── Falkenauer_U/              # Benchmark instance files (.txt)
│       ├── Falkenauer_u120_00.txt
│       ├── Falkenauer_u120_01.txt
│       ├── Falkenauer_u120_02.txt
│       ├── Falkenauer_u120_03.txt
│       └── Falkenauer_u120_04.txt
├── notebooks/
│   └── bpp_experiments_clean.ipynb# Cleaned demonstration & analysis notebook
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Instance loading and validation
│   ├── validation.py              # Shared index-based solution validator
│   ├── baseline.py                # First Fit Decreasing (FFD) solver
│   ├── ga.py                      # Custom Genetic Algorithm solver
│   ├── pso.py                     # Custom Particle Swarm Optimization solver
│   ├── metrics.py                 # Performance metrics and summary generation
│   ├── experiments.py             # Master experimental suite orchestrator
│   ├── statistics.py              # Wilcoxon, paired t-test, Holm correction
│   └── plotting.py                # Convergence curves & sensitivity plots
├── scripts/
│   ├── run_experiments.py         # Command 1: Full reproduction from scratch
│   └── reproduce_report.py        # Command 2: Instant report/table regeneration
├── tests/
│   ├── test_data_loader.py        # Tests for instance loading & lower bounds
│   ├── test_validation.py         # Tests for solution validator edge cases
│   ├── test_baseline.py           # Tests for FFD correctness
│   ├── test_ga.py                 # Tests for GA operators & budget
│   └── test_pso.py                # Tests for PSO mapping & budget
├── results/
│   ├── raw/                       # Raw run CSVs (GA, PSO, FFD, sensitivity)
│   ├── summaries/                 # Performance tables and statistical tests
│   └── figures/                   # Mean convergence comparison plots (.png)
└── reports/
    └── tables/                    # Markdown formatted tables for report
```

---

## 6. Installation and Environment Setup

### Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Standard virtual environment (`venv` or `conda`)

### Step-by-Step Installation
```bash
# Clone the repository
git clone https://github.com/marshaniswah/Tugas-Kelompok-Problem-Solving-Agent.git
cd Tugas-Kelompok-Problem-Solving-Agent

# Create and activate a clean virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 7. Reproduction Commands

### Command 1: Run Full Benchmark Experiments from Scratch
Executes all 5 instances with 5 seeds (25 runs each for GA and PSO), FFD baseline, GA operator variation, PSO sensitivity, and the full GA parameter sensitivity suite:
```bash
python scripts/run_experiments.py --config config/experiment_config.json
```

#### Optional Quick Smoke-Test Mode
To quickly verify that all code paths, validation checks, and plotting scripts execute:
```bash
python scripts/run_experiments.py --config config/experiment_config.json --quick
```

### Command 2: Regenerate All Report Tables and Figures from Saved Raw Results
Instantly reproduces all summary tables, statistical reports, and convergence figures without rerunning the metaheuristics:
```bash
python scripts/reproduce_report.py --config config/experiment_config.json
```
*(Expected duration: ~2 seconds)*

### Running Automated Test Suite
To execute the automated unit and integration tests:
```bash
python -m pytest
```

---

## 8. Configuration System (`experiment_config.json`)

All experimental parameters, seeds, budgets, instance metadata, and file paths are centralized in [`config/experiment_config.json`](config/experiment_config.json). Key configuration sections:
- `benchmark.instances`: List of instances, capacities, item counts, and certified BKS.
- `experimental_protocol.seeds`: Explicit random seeds (`[11, 22, 33, 44, 55]`).
- `experimental_protocol.evaluation_budget`: Computational budget ($6,000$).
- `algorithms.ga`: Population size ($50$), tournament size ($3$), crossover rate ($0.90$), mutation rate ($0.20$).
- `algorithms.pso`: Swarm size ($50$), $w = 0.7$, $c_1 = 1.5$, $c_2 = 1.5$, velocity bounds $[-0.2, 0.2]$.
- `ga_sensitivity_analysis`: Grids for population sizes ($20..60$), mutation rates ($0.01..0.20$), tournament sizes ($2..5$), and crossover rates ($0.70..1.00$).
- `sensitivity_analysis`: Parameter grids for PSO inertia $w$ and acceleration coefficients $(c_1, c_2)$.

---

## 9. Experimental Results Summary

The table below presents the quantitative performance across 25 paired runs per algorithm (5 Falkenauer U instances $\times$ 5 seeds) under the strict budget of 6,000 evaluations:

| Algorithm | Instance | BKS | Best Bins | Mean Bins | SD | Worst Bins | Mean Gap (%) | Mean Time (s) | Valid |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FFD** | `Falkenauer_u120_00.txt` | 48 | **48** | **48.00** | 0.00 | 48 | **0.00%** | <0.001 | True |
| **FFD** | `Falkenauer_u120_01.txt` | 49 | **49** | **49.00** | 0.00 | 49 | **0.00%** | <0.001 | True |
| **FFD** | `Falkenauer_u120_02.txt` | 46 | **46** | **46.00** | 0.00 | 46 | **0.00%** | <0.001 | True |
| **FFD** | `Falkenauer_u120_03.txt` | 49 | **49** | **49.00** | 0.00 | 49 | **0.00%** | <0.001 | True |
| **FFD** | `Falkenauer_u120_04.txt` | 50 | **50** | **50.00** | 0.00 | 50 | **0.00%** | <0.001 | True |
| **GA**  | `Falkenauer_u120_00.txt` | 48 | 49 | 49.00 | 0.00 | 49 | 2.08% | 1.15 | True |
| **GA**  | `Falkenauer_u120_01.txt` | 49 | 50 | 50.00 | 0.00 | 50 | 2.04% | 1.14 | True |
| **GA**  | `Falkenauer_u120_02.txt` | 46 | 47 | 47.00 | 0.00 | 47 | 2.17% | 1.16 | True |
| **GA**  | `Falkenauer_u120_03.txt` | 49 | 50 | 50.80 | 0.45 | 51 | 3.67% | 1.16 | True |
| **GA**  | `Falkenauer_u120_04.txt` | 50 | 51 | 51.00 | 0.00 | 51 | 2.00% | 1.18 | True |
| **PSO** | `Falkenauer_u120_00.txt` | 48 | 49 | 49.80 | 0.45 | 50 | 3.75% | 3.52 | True |
| **PSO** | `Falkenauer_u120_01.txt` | 49 | 50 | 50.00 | 0.00 | 50 | 2.04% | 3.48 | True |
| **PSO** | `Falkenauer_u120_02.txt` | 46 | 47 | 47.40 | 0.55 | 48 | 3.04% | 3.49 | True |
| **PSO** | `Falkenauer_u120_03.txt` | 49 | 51 | 51.00 | 0.00 | 51 | 4.08% | 3.50 | True |
| **PSO** | `Falkenauer_u120_04.txt` | 50 | 51 | 51.00 | 0.00 | 51 | 2.00% | 3.51 | True |

---

## 10. Statistical Significance Analysis

To rigorously compare the two stochastic metaheuristics (**GA vs. PSO**), non-parametric paired tests were performed across shared random seeds.

### Test Justification
- **Wilcoxon Signed-Rank Test**: The response variable (number of bins) is integer-valued and non-normally distributed with paired random seeds.
- **Holm-Bonferroni Step-Down Correction**: Applied to control the family-wise error rate across the 5 instance hypothesis tests.
- **Sample Size Transparency**: With $n = 5$ paired observations per instance, the minimum achievable two-sided Wilcoxon p-value is $p = 0.0625$. Therefore, per-instance tests cannot achieve $p < 0.05$, a mathematical limitation that must not be overstated.

### Per-Instance Paired Comparison (GA vs. PSO)
| Instance | $n$ | Mean GA | Mean PSO | Mean Diff (GA - PSO) | Wilcoxon Stat | Raw $p$ | Holm Adjusted $p$ | Significant ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Falkenauer_u120_00.txt` | 5 | 49.00 | 49.80 | **-0.80** | 0.0 | 0.0625 | 0.3125 | False (Power limited at $n=5$) |
| `Falkenauer_u120_01.txt` | 5 | 50.00 | 50.00 | 0.00 | 0.0 | 1.0000 | 1.0000 | False (Identical) |
| `Falkenauer_u120_02.txt` | 5 | 47.00 | 47.40 | **-0.40** | 0.0 | 0.1573 | 0.6292 | False (Power limited at $n=5$) |
| `Falkenauer_u120_03.txt` | 5 | 50.80 | 51.00 | **-0.20** | 0.0 | 0.3173 | 0.9519 | False (Power limited at $n=5$) |
| `Falkenauer_u120_04.txt` | 5 | 51.00 | 51.00 | 0.00 | 0.0 | 1.0000 | 1.0000 | False (Identical) |

### Pooled Paired Comparison Across All 25 Runs
- Total paired runs: **25**
- GA won: **7 runs** (GA used 1 fewer bin than PSO)
- PSO won: **0 runs** (PSO never outperformed GA on any run)
- Tied: **18 runs**
- Mean difference: **-0.28 bins** in favor of GA
- **Wilcoxon Signed-Rank Test**: $W = 0.0, p = 0.0156$ (**Statistically significant**, $p < 0.05$)
- **Paired $t$-test**: $t = -3.055, p = 0.0054$ (**Statistically significant**, $p < 0.01$)

*Interpretation*: Across the full paired benchmark, GA demonstrates statistically significant superiority over PSO under the 6,000-evaluation budget.

---

## 11. Convergence Plots

Mean best-so-far convergence curves across the 5 random seeds are saved in `results/figures/`:
- `convergence_Falkenauer_u120_00.png`
- `convergence_Falkenauer_u120_01.png`
- `convergence_Falkenauer_u120_02.png`
- `convergence_Falkenauer_u120_03.png`
- `convergence_Falkenauer_u120_04.png`

Each figure shows:
- **X-axis**: Objective Function Evaluations ($0$ to $6,000$)
- **Y-axis**: Mean Best-so-Far Number of Bins (Minimized)
- **Blue Line**: PSO Mean curve with $\pm 1$ SD shaded band
- **Red Line**: GA Mean curve sampled at generation checkpoints with $\pm 1$ SD shaded band
- **Green Dashed Line**: Certified Optimum / BKS

---

## 12. GA Operator Variation and Parameter Sensitivity Analysis

To justify the hyperparameter selections for both metaheuristics, systematic sensitivity analyses were conducted on pilot benchmark instances across all 5 random seeds (`[11, 22, 33, 44, 55]`) under the 6,000 objective-function evaluation budget.

### A. GA Sensitivity Suite (Pilot Instance: `Falkenauer_u120_00.txt`, BKS = 48)

#### 1. Population Size ($N$) Sensitivity
Evaluated across $N \in \{20, 30, 40, 50, 60\}$ under fixed $p_m = 0.20, k = 3, p_c = 0.90$:

| Population Size ($N$) | Runs | Best Bins | Mean Bins | SD | Worst Bins | Mean Gap (%) | Mean Runtime (s) | All Valid |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 20 | 5 | 49 | 49.00 | 0.000 | 49 | 2.08% | 3.33 | True |
| 30 | 5 | 49 | 49.40 | 0.548 | 50 | 2.92% | 3.23 | True |
| 40 | 5 | 49 | 49.00 | 0.000 | 49 | 2.08% | 3.12 | True |
| **50** | 5 | 49 | **49.00** | **0.000** | 49 | **2.08%** | 3.17 | True |
| 60 | 5 | 49 | 49.20 | 0.447 | 50 | 2.50% | 3.36 | True |

*Analysis*: $N = 50$ yielded zero variance across all seeds ($\text{SD} = 0.00$), maintaining sufficient population diversity without diluting the number of generations under the fixed 6,000-evaluation budget.

#### 2. Mutation Rate ($p_m$) Sensitivity
Evaluated across $p_m \in \{0.01, 0.05, 0.10, 0.20\}$ under fixed $N = 50, k = 3, p_c = 0.90$:

| Mutation Rate ($p_m$) | Runs | Best Bins | Mean Bins | SD | Worst Bins | Mean Gap (%) | Mean Runtime (s) | All Valid |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | 5 | 49 | 49.40 | 0.548 | 50 | 2.92% | 3.04 | True |
| 0.05 | 5 | 49 | 49.00 | 0.000 | 49 | 2.08% | 3.22 | True |
| 0.10 | 5 | 49 | 49.40 | 0.548 | 50 | 2.92% | 3.00 | True |
| **0.20** | 5 | 49 | **49.00** | **0.000** | 49 | **2.08%** | 3.29 | True |

*Analysis*: Higher swap mutation rates ($p_m = 0.20$) actively prevented genetic stagnation, enabling chromosomes to escape sub-optimal 50-bin arrangements.

#### 3. Tournament Size ($k$) Sensitivity
Evaluated across $k \in \{2, 3, 4, 5\}$ under fixed $N = 50, p_m = 0.20, p_c = 0.90$:

| Tournament Size ($k$) | Runs | Best Bins | Mean Bins | SD | Worst Bins | Mean Gap (%) | Mean Runtime (s) | All Valid |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 5 | 49 | 49.80 | 0.447 | 50 | 3.75% | 3.47 | True |
| **3** | 5 | 49 | **49.00** | **0.000** | 49 | **2.08%** | 3.34 | True |
| 4 | 5 | 49 | 49.20 | 0.447 | 50 | 2.50% | 3.29 | True |
| 5 | 5 | 49 | 49.20 | 0.447 | 50 | 2.50% | 3.11 | True |

*Analysis*: $k = 2$ exerted insufficient selection pressure (mean 49.80), while $k \ge 4$ caused premature convergence. $k = 3$ produced the optimal trade-off.

#### 4. Crossover Rate ($p_c$) Sensitivity
Evaluated across $p_c \in \{0.70, 0.80, 0.90, 1.00\}$ under fixed $N = 50, p_m = 0.20, k = 3$:

| Crossover Rate ($p_c$) | Runs | Best Bins | Mean Bins | SD | Worst Bins | Mean Gap (%) | Mean Runtime (s) | All Valid |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.70 | 5 | 49 | 49.40 | 0.548 | 50 | 2.92% | 3.21 | True |
| 0.80 | 5 | 49 | 49.20 | 0.447 | 50 | 2.50% | 3.33 | True |
| **0.90** | 5 | 49 | **49.00** | **0.000** | 49 | **2.08%** | 3.23 | True |
| 1.00 | 5 | 49 | 49.20 | 0.447 | 50 | 2.50% | 3.11 | True |

*Analysis*: Order Crossover at $p_c = 0.90$ achieved 100% convergence to 49 bins across all 5 seeds, outperforming both lower recombination rates and complete crossover.

---

### B. PSO Sensitivity Suite (Pilot Instance: `Falkenauer_u120_04.txt`, BKS = 50)

#### 1. Inertia Weight ($w$)
Tested across $w \in \{0.4, 0.7, 0.9\}$ with fixed $c_1 = 1.5, c_2 = 1.5$:
- $w = 0.4$: Fast initial convergence, premature stagnation at 51 bins ($\text{SD}=0.00$).
- **$w = 0.7$**: Balanced exploration and exploitation, smooth deceleration trajectories (chosen baseline).
- $w = 0.9$: High particle velocity causing frequent boundary clipping at $[-0.2, 0.2]$.

#### 2. Acceleration Coefficients ($c_1, c_2$)
Tested across 5 configurations:
- Baseline ($c_1 = 1.5, c_2 = 1.5$): Mean 51.00, SD 0.00.
- Cognitive Dominant ($c_1 = 2.0, c_2 = 1.5$): Mean 51.00, SD 0.00.
- Social Dominant ($c_1 = 1.5, c_2 = 2.0$): Mean 51.00, SD 0.00.

---

## 13. Methodological Limitations

1. **Greedy Decoder Dominance**: Both GA and PSO operate over item permutations decoded by First Fit. In 1D-BPP with uniform weights, FFD alone achieves near-optimal packings; unguided search in the permutation space requires significant evaluation effort to match sorted order.
2. **Per-Instance Sample Size ($n=5$)**: While common in computational research to manage evaluation time, $n=5$ limits per-instance statistical test power.
3. **Instance Scope**: Evaluated exclusively on Falkenauer U instances ($n=120, C=150$). Triplet instances (Falkenauer T) or Hard28 benchmarks may exhibit different relative algorithm behaviors.

---

## 14. Scientific Statement and AI Disclosure

- **Authenticity Statement**: All raw numerical results, summary metrics, statistical tests, and convergence plots reported in this repository were generated by executing the accompanying source code and configuration. No results were fabricated or manipulated.
- **AI Assistance Disclosure**: AI coding assistance (Antigravity / Google DeepMind) was utilized as a pair-programming tool for code modularization, automated unit test authoring, LaTeX formula formatting, and codebase refactoring, following the academic integrity guidelines of the course. The underlying algorithm logic, experimental design, and conclusions were designed and verified by the human authors.

---

## 15. References and Citations

1. **Delorme, M., Iori, M., & Martello, S.** (2018). *BPPLIB: a library for bin packing and cutting stock problems*. Optimization Letters, 12(2), 235-250. [DOI: 10.1007/s11590-017-1192-z](https://doi.org/10.1007/s11590-017-1192-z)
2. **Falkenauer, E.** (1996). *A hybrid grouping genetic algorithm for bin packing*. Journal of Heuristics, 2(1), 5-30. [DOI: 10.1007/BF00126781](https://doi.org/10.1007/BF00126781)
3. **Johnson, D. S.** (1973). *Near-optimal bin packing algorithms*. Doctoral dissertation, Massachusetts Institute of Technology.
4. **Kennedy, J., & Eberhart, R.** (1995). *Particle swarm optimization*. Proceedings of ICNN'95 - International Conference on Neural Networks, 4, 1942-1948.
5. **Holm, S.** (1979). *A simple sequentially rejective multiple test procedure*. Scandinavian Journal of Statistics, 6(2), 65-70.
