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


def test_that_best_fitness_improves_over_generations(test_items):
    ga = KnapsackGA(
        items=test_items,
        max_weight=20,
        population_size=10,
        seed=123,
    )
    _, result = ga.run(50)
    best_fitness_over_time = [h[1] for h in result.history]
    assert best_fitness_over_time[-1] >= best_fitness_over_time[0]


def test_that_avg_fitness_improves_over_generations(test_items):
    ga = KnapsackGA(
        items=test_items,
        max_weight=20,
        population_size=10,
        seed=123,
    )
    _, result = ga.run(50)
    avg_fitness_over_time = [h[2] for h in result.history]
    assert avg_fitness_over_time[-1] >= avg_fitness_over_time[0]


def test_that_run_is_deterministic_with_seed(test_items):
    ga1 = KnapsackGA(test_items, max_weight=20, population_size=10, seed=999)
    ga2 = KnapsackGA(test_items, max_weight=20, population_size=10, seed=999)

    n_generations = 50
    _, result1 = ga1.run(n_generations)
    _, result2 = ga2.run(n_generations)

    assert result1.best_individual == result2.best_individual
    assert result1.best_fitness == result2.best_fitness
