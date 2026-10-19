# Benchmark Dataset: Falkenauer U Instances

## 1. Overview and Provenance
This directory contains five benchmark instances from the classic Falkenauer U (Uniform) problem set for the One-Dimensional Bin Packing Problem (1D-BPP).

- **Benchmark Suite**: Falkenauer U (Uniform distribution)
- **Author**: Emanuel Falkenauer (1996)
- **Original Publication**:
  - Falkenauer, E. (1996). *A hybrid grouping genetic algorithm for bin packing*. Journal of Heuristics, 2(1), 5-30. DOI: [10.1007/BF00126781](https://doi.org/10.1007/BF00126781)
- **Benchmark Source Repositories**:
  - **BPPLIB GitHub Repository**: [https://github.com/mdelorme2/BPPLIB](https://github.com/mdelorme2/BPPLIB) (Delorme, Iori, & Martello, 2018)
  - **BPPLIB Web Library (University of Bologna)**: [https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library](https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library)
  - Reference: Delorme, M., Iori, M., & Martello, S. (2018). *BPPLIB: a library for bin packing and cutting stock problems*. Optimization Letters, 12(2), 235-250.

## 2. File Format Specification
Each instance is stored as a plain text (`.txt`) file formatted as follows:
- **Line 1**: Number of items $n$ (integer, $n = 120$ for the `u120` series)
- **Line 2**: Bin capacity $C$ (integer, $C = 150$)
- **Lines 3 to $n+2$**: Individual item weights $w_i$ (integers drawn uniformly from $[20, 100]$)

## 3. Instance Properties and Proven Best-Known Solutions (BKS)
The optimal solution (or Best-Known Solution) for all five selected instances has been mathematically proven and certified in BPPLIB (see `Solutions.xlsx`, Sheet `Falkenauer`, status: `Solved`).

| Instance Filename | Items ($n$) | Bin Capacity ($C$) | Total Weight ($\sum w_i$) | Continuous Volume Lower Bound $\lceil \frac{\sum w_i}{C} \rceil$ | Proven Optimum / BKS | Status in BPPLIB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Falkenauer_u120_00.txt` | 120 | 150 | 7,078 | 48 | **48** | Solved |
| `Falkenauer_u120_01.txt` | 120 | 150 | 7,205 | 49 | **49** | Solved |
| `Falkenauer_u120_02.txt` | 120 | 150 | 6,794 | 46 | **46** | Solved |
| `Falkenauer_u120_03.txt` | 120 | 150 | 7,285 | 49 | **49** | Solved |
| `Falkenauer_u120_04.txt` | 120 | 150 | 7,354 | 50 | **50** | Solved |

## 4. Gap Formula
The relative percentage gap between the objective value achieved by an algorithm ($z$) and the sourced BKS is calculated as:
$$\text{gap\_percent} = \frac{z - \text{BKS}}{\text{BKS}} \times 100\%$$
A gap of $0.00\%$ indicates that the algorithm found a proven optimal packing.
