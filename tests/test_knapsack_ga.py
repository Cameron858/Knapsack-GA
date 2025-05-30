from knapsack import KnapsackGA
import pytest


@pytest.mark.parametrize(
    "rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
def test_that_knapsack_doesnt_raise_value_error_for_valid_crossover_rates(rate):
    KnapsackGA([], 0, 10, crossover_rate=rate)


@pytest.mark.parametrize(
    "rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
def test_that_knapsack_doesnt_raise_value_error_for_valid_mutation_rates(rate):
    KnapsackGA([], 0, 10, mutation_rate=rate)


@pytest.mark.parametrize(
    "rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
def test_that_knapsack_doesnt_raise_value_error_for_valid_elitism_rates(rate):
    KnapsackGA([], 0, 10, elitism_rate=rate)


invalid_rates = [
    -1.0,  # below 0
    -0.1,  # slightly below 0
    1.1,  # slightly above 1
    2.0,  # above 1
    100,  # way above 1
    -100,  # way below 0
    float("inf"),  # positive infinity
    float("-inf"),  # negative infinity
    float("nan"),  # not a number
]


@pytest.mark.parametrize("rate", invalid_rates)
def test_that_knapsack_raises_value_error_for_invalid_crossover_rates(rate):

    with pytest.raises(ValueError, match="Crossover rate must be between 0 and 1."):
        KnapsackGA([], 0, 10, crossover_rate=rate)


@pytest.mark.parametrize("rate", invalid_rates)
def test_that_knapsack_raises_value_error_for_invalid_mutation_rates(rate):
    with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1."):
        KnapsackGA([], 0, 10, mutation_rate=rate)


@pytest.mark.parametrize("rate", invalid_rates)
def test_that_knapsack_raises_value_error_for_invalid_elitism_rates(rate):
    with pytest.raises(ValueError, match="Elitism rate must be between 0 and 1."):
        KnapsackGA([], 0, 10, elitism_rate=rate)


def test_that_seed_creates_reproducible_results():
    # arbitary, as long as both instances get the same seed
    seed = 19

    ga1 = KnapsackGA([], 0, 10, seed=seed)
    ga2 = KnapsackGA([], 0, 10, seed=seed)

    pop1 = ga1._generate_population()
    pop2 = ga2._generate_population()

    assert pop1 == pop2
def test_that_generate_individual_has_correct_length_and_binary_values(test_items):
    ga = KnapsackGA(items=test_items, max_weight=20, population_size=10, seed=42)
    individual = ga._generate_individual()

    assert len(individual) == len(test_items)
    assert all(gene in (0, 1) for gene in individual)
