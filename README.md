# Knapsack-GA

A Python implementation of the 0/1 Knapsack Problem solved using a Genetic Algorithm (GA).

## Overview

This project applies a genetic algorithm to solve the classic 0/1 knapsack problem, where each item can be included (1) or excluded (0) from the knapsack. The goal is to maximize total value without exceeding the maximum weight capacity.

Each potential solution (individual) is encoded as a binary chromosome, where each bit corresponds to whether an item is included (1) or not (0).

### Algorithm Flowchart
<img src="docs/algorithm.svg" width="600" height="600" alt="Flowchart illustrating the steps of a genetic algorithm for solving the 0/1 knapsack problem. The process starts with initializing GA parameters, generating the initial population, and evaluating the fitness of each individual. It checks if the stopping condition is met. If not, it selects parents via tournament, applies crossover to produce offspring, applies mutation to offspring, and evaluates the fitness of offspring. The new population is formed and the process repeats. If the stopping condition is met, the best solution is returned and the algorithm ends. The flowchart includes decision points labeled Yes and No, and all process steps are clearly labeled with descriptive text. The overall tone is instructional and methodical, supporting understanding of the genetic algorithm workflow." />

Key Learnings
- Gained a solid understanding of genetic algorithm components such as representation, fitness evaluation, selection, crossover, mutation, and elitism.
- Implemented GA operators from scratch in Python, emphasising modular and clear code structure.
- Explored the balance between exploration and exploitation during the evolutionary search process.
- Compared brute-force and heuristic approaches, highlighting the efficiency benefits of genetic algorithms for combinatorial optimisation problems like knapsack.
- Applied tournament selection and single-point crossover techniques to evolve the population effectively.
- Developed tools to track performance and convergence over generations with meaningful visualisations.
- Leveraged Python's data and plotting libraries to manage experiments and present results clearly.
- Understood the strengths and limitations of genetic algorithms for NP-hard problems, motivating further study into multi-objective optimisation and hybrid approaches.
- Basics of using `uv` for dependency management

### Example
```python
# Items 2, 4 and 5 are chosen to be in the knapsack
solution = [0, 1, 0, 1, 1]
```

## Features
- Binary-encoded representation of solutions
- Fitness based on total value, penalising overweight solutions
- Configurable population size, mutation rate, crossover method, etc.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Cameron858/Knapsack-GA
    ```

2. Navigate to the project directory:
    ```bash
    cd knapsack-ga
    ```

3. Install [uv](https://github.com/astral-sh/uv) if not already installed:
    ```bash
    pipx install uv
    ```

4. Install the required dependencies using uv:
    ```bash
    uv pip install -r pyproject.toml
    ```

5. (Optional) Run tests to verify the installation:
    ```bash
    uv run pytest
    ```

6. (Optional) Activate a virtual environment (recommended):
    ```bash
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate # On macOS/Linux
    ```

## Usage Example

```python
from knapsack.algorithm import KnapsackGA, Item

# Define items
items = [
    Item(name="A", value=10, weight=5),
    Item(name="B", value=7, weight=3),
    Item(name="C", value=12, weight=8),
    Item(name="D", value=8, weight=4),
]

# Initialize GA
knapsack = KnapsackGA(
    items=items,
    max_weight=10,
    population_size=30,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism_rate=0.1,
)

# Run the algorithm
selected_items, result = knapsack.run(generations=50)

print("Best value:", result.best_fitness)
print("Selected items:", [item.name for item in selected_items])
print(f"Runtime: {result.runtime:.3f} seconds for {result.generations} generations")

# result.history contains (generation, best_fitness, avg_fitness) for each generation

## Output
- `selected_items`: List of `Item` objects chosen in the best solution
- `result`: `GAResult` object with:
  - `best_individual`: The best solution as a binary list
  - `best_fitness`: Total value of the best solution
  - `history`: List of tuples `(generation, best_fitness, average_fitness)` for each generation
  - `runtime`: Total runtime in seconds
  - `generations`: Number of generations run
```

## Parameters
- `items`: List of `Item` objects (each with `name`, `value`, `weight`)
- `max_weight`: Maximum total weight allowed in the knapsack
- `population_size`: Number of individuals in each generation
- `crossover_rate`: Probability of crossover between parents (default: 0.8)
- `mutation_rate`: Probability of mutating each gene (default: 0.1)
- `elitism_rate`: Fraction of top individuals carried to next generation (default: 0.1)
- `generations`: Number of generations to run (default: 50)

## Customisation
- You can modify the list of items, knapsack capacity, and GA parameters to suit your problem.
- The algorithm uses tournament selection, single-point crossover, mutation, and elitism.

## Future Work
- **Alternative operators:** Experiment with different selection, crossover, or mutation strategies.
- **Constraint variations:** Add support for additional constraints (e.g., grouped items, minimum/maximum item counts).
- **Performance improvements:** Profile and parallelize the code for larger problem instances.
- **Visualization:** Enhance result and convergence visualisations, or add interactive dashboards.
- **Benchmarking:** Compare GA results with other algorithms (e.g., simulated annealing, particle swarm, or exact solvers).
- **Experiment tracking:** Automate experiment management and result logging for reproducibility.

## Dependencies

This project was written in python 3.13. Project dependencies can be found in `pyproject.toml`.

## Source Code
See `src/knapsack/algorithm.py` for implementation details.

P.S: I do *usually* use proper branch management i.e. GitFlow, but for a project like this I just wanted to code :) 