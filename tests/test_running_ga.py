from knapsack import KnapsackGA
import pytest

from knapsack.algorithm import Item


def test_that_algorithm_runs_with_standard_params(test_items):
    """Run basic smoke test."""
    ga = KnapsackGA(
        items=test_items,
        max_weight=20,
        population_size=50,
        seed=19,
    )

    solution, result = ga.run(generations=50)
    assert solution  # has items
    assert all(isinstance(item, Item) for item in solution)
    assert hasattr(result, "best_individual")
    assert hasattr(result, "best_fitness")
    assert hasattr(result, "history")
    assert hasattr(result, "runtime")
    assert hasattr(result, "generations")


def test_that_each_generation_has_recorded_history(test_items):
    n_generations = 50
    ga = KnapsackGA(
        items=test_items,
        max_weight=20,
        population_size=50,
        seed=19,
    )
    solution, result = ga.run(generations=n_generations)

    assert len(result.history) == n_generations, "History is incorrect length."
    assert all(
        len(h) == 3 for h in result.history
    ), "History contains mismatched record lengths."
