# Knapsack-GA

A Python implementation of the 0/1 Knapsack Problem solved using a Genetic Algorithm (GA).

## Overview

This project applies a genetic algorithm to solve the classic 0/1 knapsack problem, where each item can be included (1) or excluded (0) from the knapsack. The goal is to maximize total value without exceeding the maximum weight capacity.

Each potential solution (individual) is encoded as a binary chromosome, where each bit corresponds to whether an item is included (1) or not (0).

### Example
```python
# Items 2, 4 and 5 are chosen to be in the knapsack
solution = [0, 1, 0, 1, 1]
```

## Features
- Binary-encoded representation of solutions
- Fitness based on total value, penalising overweight solutions
- Configurable population size, mutation rate, crossover method, etc.