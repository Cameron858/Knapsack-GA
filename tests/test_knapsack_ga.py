from knapsack import KnapsackGA
import pytest


def test_that_init_raises_value_error_for_empty_items():
    with pytest.raises(
        ValueError, match="`items` must be a non-empty list of Item objects."
    ):
        KnapsackGA(items=[], max_weight=10, population_size=5)


def test_that_init_raises_value_error_for_invalid_item_type():
    with pytest.raises(
        ValueError, match="`items` must be a non-empty list of Item objects."
    ):
        KnapsackGA(items=["not", "items"], max_weight=10, population_size=5)


@pytest.mark.parametrize(
    "rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
def test_that_knapsack_doesnt_raise_value_error_for_valid_crossover_rates(
    rate, test_items
):
    KnapsackGA(test_items, 0, 10, crossover_rate=rate)


@pytest.mark.parametrize(
    "rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
def test_that_knapsack_doesnt_raise_value_error_for_valid_mutation_rates(
    rate, test_items
):
    KnapsackGA(test_items, 0, 10, mutation_rate=rate)


@pytest.mark.parametrize(
    "rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
def test_that_knapsack_doesnt_raise_value_error_for_valid_elitism_rates(
    rate, test_items
):
    KnapsackGA(test_items, 0, 10, elitism_rate=rate)


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
def test_that_knapsack_raises_value_error_for_invalid_crossover_rates(rate, test_items):

    with pytest.raises(ValueError, match="Crossover rate must be between 0 and 1."):
        KnapsackGA(test_items, 0, 10, crossover_rate=rate)


@pytest.mark.parametrize("rate", invalid_rates)
def test_that_knapsack_raises_value_error_for_invalid_mutation_rates(rate, test_items):
    with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1."):
        KnapsackGA(test_items, 0, 10, mutation_rate=rate)


@pytest.mark.parametrize("rate", invalid_rates)
def test_that_knapsack_raises_value_error_for_invalid_elitism_rates(rate, test_items):
    with pytest.raises(ValueError, match="Elitism rate must be between 0 and 1."):
        KnapsackGA(test_items, 0, 10, elitism_rate=rate)


def test_that_seed_creates_reproducible_results(test_items):
    # arbitary, as long as both instances get the same seed
    seed = 19

    ga1 = KnapsackGA(test_items, 0, 10, seed=seed)
    pop1 = ga1._generate_population()

    ga2 = KnapsackGA(test_items, 0, 10, seed=seed)
    pop2 = ga2._generate_population()

    assert pop1 == pop2


def test_that_generate_individual_has_correct_length_and_binary_values(test_items):
    ga = KnapsackGA(items=test_items, max_weight=20, population_size=10, seed=19)
    individual = ga._generate_individual()

    assert len(individual) == len(test_items)
    assert all(gene in (0, 1) for gene in individual)


def test_that_generate_population_has_correct_size_and_valid_individuals(test_items):
    ga = KnapsackGA(items=test_items, max_weight=20, population_size=5, seed=19)
    population = ga._generate_population()

    assert len(population) == ga.population_size

    for individual in population:
        assert len(individual) == len(test_items)
        assert all(gene in (0, 1) for gene in individual)


@pytest.mark.parametrize(
    "individual, expected_value, max_weight",
    [
        ([0, 0, 0, 0, 0], 0, 100),  # arbitarily large max_weight
        ([0, 1, 1, 0, 0], 20, 100),  # arbitarily large max_weight
        ([1, 1, 1, 1, 1], 0, 5),  # weight < max weight of test items
        ([1, 1, 1, 1, 1], 43, 23),  # weight == max weight
    ],
)
def test_that_evaluating_an_individual_returns_correct_values(
    individual, expected_value, max_weight, test_items
):
    ga = KnapsackGA(items=test_items, max_weight=max_weight, population_size=5)

    value = ga.evaluate(individual)
    assert value == expected_value


def test_that_mutating_with_zero_rate_produces_no_mutation(test_items):
    ga = KnapsackGA(test_items, 0, 10, mutation_rate=0)

    ind_pre_mutation = ga._generate_individual()

    ind_post_mutation = ga.mutate(ind_pre_mutation)

    assert ind_post_mutation == ind_pre_mutation


def test_that_mutating_with_one_rate_always_produces_mutation(test_items):
    ga = KnapsackGA(test_items, 0, 10, mutation_rate=1)

    ind_pre_mutation = ga._generate_individual()
    expected_individual = [1 - bit for bit in ind_pre_mutation]

    ind_post_mutation = ga.mutate(ind_pre_mutation)

    assert ind_post_mutation == expected_individual


def test_that_mutation_applies_based_on_probability(mocker, test_items):
    ga = KnapsackGA(test_items, 0, 10, mutation_rate=0.5)
    ind_pre_mutation = [1, 1, 0, 0, 1]

    # Patch to return values causing mutation only for index 1 and 3
    mocker.patch("random.random", side_effect=[0.6, 0.4, 0.8, 0.3, 0.9])

    ind_post_mutation = ga.mutate(ind_pre_mutation)
    expected = [1, 0, 0, 1, 1]

    assert ind_post_mutation == expected


def test_that_mutating_invalid_individual_raises_value_error(test_items):
    ga = KnapsackGA(test_items, 0, 10, mutation_rate=1)
    invalid_individual = [0, 2, 1]  # 2 is invalid

    with pytest.raises(ValueError, match="Bit value needs to be 0 or 1"):
        ga.mutate(invalid_individual)


def test_that_mutate_does_not_modify_input_in_place(test_items):
    ga = KnapsackGA(test_items, 0, 10, mutation_rate=1)
    ind_pre_mutation = [0, 1, 0, 1]
    ind_copy = ind_pre_mutation[:]

    ga.mutate(ind_pre_mutation)

    assert ind_pre_mutation == ind_copy  # ensure no in-place change


def test_that_single_point_crossover_produces_valid_children(mocker, test_items):
    parent1 = [0, 0, 0, 0, 0]
    parent2 = [1, 1, 1, 1, 1]
    ga = KnapsackGA(test_items, 0, 10)

    # Mock the crossover point to a known value
    mocker.patch("random.randint", return_value=2)

    child1, child2 = ga.single_point_crossover(parent1, parent2)

    assert child1 == [0, 0, 1, 1, 1]
    assert child2 == [1, 1, 0, 0, 0]


def test_that_single_point_crossover_with_split_at_one(mocker, test_items):
    parent1 = [1, 1, 1]
    parent2 = [0, 0, 0]
    ga = KnapsackGA(test_items, 0, 10)

    mocker.patch("random.randint", return_value=1)

    child1, child2 = ga.single_point_crossover(parent1, parent2)

    assert child1 == [1, 0, 0]
    assert child2 == [0, 1, 1]


def test_that_single_point_crossover_with_split_at_end_minus_one(mocker, test_items):
    parent1 = [1, 0, 0]
    parent2 = [0, 1, 1]
    ga = KnapsackGA(test_items, 0, 10)

    mocker.patch("random.randint", return_value=2)

    child1, child2 = ga.single_point_crossover(parent1, parent2)

    assert child1 == [1, 0, 1]
    assert child2 == [0, 1, 0]


def test_that_crossover_raises_error_on_unequal_length(test_items):
    parent1 = [1, 1, 1]
    parent2 = [0, 0]
    ga = KnapsackGA(test_items, 0, 10)

    with pytest.raises(AssertionError, match="Parents must be of equal length"):
        ga.single_point_crossover(parent1, parent2)


def test_that_tournament_returns_highest_fitness_individual(mocker, test_items):
    ga = KnapsackGA(test_items, max_weight=100, population_size=10)
    population = [
        [0, 0, 0, 0, 0],  # value 0
        [0, 0, 0, 0, 0],  # value 0
        [0, 0, 0, 0, 0],  # value 0
        [0, 0, 0, 0, 0],  # value 0
        [1, 0, 1, 0, 0],  # value 25
    ]

    mocker.patch("random.sample", return_value=population)

    winner = ga.tournament(population, k=5)
    assert winner == [1, 0, 1, 0, 0]  # highest value: 25


def test_that_tournament_raises_value_error_with_k_greater_than_population(test_items):
    ga = KnapsackGA(test_items, max_weight=100, population_size=5)
    population = [[0, 1, 0, 0, 0]] * 3

    with pytest.raises(ValueError, match="k=5 is greater than population size=3"):
        ga.tournament(population, k=5)


def test_that_tournament_with_k_smaller_than_population_returns_population_member(
    test_items,
):
    ga = KnapsackGA(test_items, max_weight=100, population_size=10)
    population = [[0, 0, 1, 0, 0]] * 10  # all the same

    winner = ga.tournament(population, k=3)
    assert winner in population


@pytest.mark.parametrize(
    "individual", [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
)
def test_that_decoding_an_individual_returns_correct_items(individual, test_items):
    ga = KnapsackGA(test_items, 100, 50)

    items = ga.decode_individual(individual)

    expected_items = [
        item for item, selected in zip(test_items, individual) if selected
    ]

    assert items == expected_items
