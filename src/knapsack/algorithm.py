from dataclasses import dataclass
import random
from typing import Literal

type Individual = list[Literal[0, 1]]


@dataclass
class Item:
    name: str
    value: float
    weight: float


class KnapsackGA:
    def __init__(
        self,
        items: list[Item],
        max_weight: float,
        population_size: int,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
    ):
        self.items = items
        self.num_items = len(items)
        self.max_weight = max_weight

        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_items={self.num_items}, "
            f"max_weight={self.max_weight}, "
            f"population_size={self.population_size}, "
            f"crossover_rate={self.crossover_rate:.2f}, "
            f"mutation_rate={self.mutation_rate:.2f}, "
            f"elitism_rate={self.elitism_rate:.2f})"
        )

    def _generate_individual(self) -> Individual:
        """
        Create a random individual for the genetic algorithm population.

        An individual is represented as a list of binary genes (0 or 1),
        where each element indicates whether the corresponding item is excluded (0) or included (1).
        The list length equals the number of items available.

        Returns
        -------
        Individual
            List of 0s and 1s representing item selection.
        """
        return [random.randint(0, 1) for _ in range(self.num_items)]

    def _generate_population(self) -> list[Individual]:
        """
        Create a population of random individuals.

        Returns
        -------
        list[Individual]
            List of randomly generated individuals.
        """
        return [self._generate_individual() for _ in range(self.population_size)]

    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate the fitness of an individual.

        The fitness is the total value of selected items, unless the total weight
        exceeds the knapsack capacity, in which case the fitness is 0.

        Parameters
        ----------
        individual : Individual
            A list of binary genes (0 or 1) representing the individual.

        Returns
        -------
        float
            The total value of selected items, or 0 if overweight.
        """
        weight = sum(
            item.weight for item, selected in zip(self.items, individual) if selected
        )

        if weight > self.max_weight:
            return 0.0

        return sum(
            item.value for item, selected in zip(self.items, individual) if selected
        )

    def mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual's genes independently with a probability defined by mutation_rate.

        Each gene (bit) in the individual has a chance to be flipped:
        0 becomes 1, and 1 becomes 0.

        Parameters
        ----------
        individual : Individual
            A list of binary genes (0 or 1) representing the individual.

        Returns
        -------
        Individual
            A new list representing the mutated individual.

        Raises
        ------
        ValueError
            If any gene in the individual is not 0 or 1.
        """

        def flip_bit(bit):
            if bit not in (0, 1):
                raise ValueError(f"Bit value needs to be 0 or 1, got {bit}")
            return 1 - bit  # simpler flip logic

        return [
            flip_bit(bit) if random.random() < self.mutation_rate else bit
            for bit in individual
        ]

    def single_point_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual]:
        """
        Perform single-point crossover between two parent individuals.

        A random crossover point is selected (excluding the ends), and
        two offspring are created by exchanging bits.

        Parameters
        ----------
        parent1 : Individual
            A list of binary genes (0 or 1) representing the individual.

        parent2 : Individual
            A list of binary genes (0 or 1) representing the individual.

        Returns
        -------
        tuple[Individual, Individual]
            Two offspring individuals resulting from crossover.
        """
        assert len(parent1) == len(parent2), "Parents must be of equal length"

        split_index = random.randint(1, len(parent1) - 1)

        child1 = parent1[:split_index] + parent2[split_index:]
        child2 = parent2[:split_index] + parent1[split_index:]

        return child1, child2

    def tournament(self, population: list[Individual], k: int = 5) -> Individual:
        """
        Select the fittest individual from a random tournament bracket.

        Randomly samples `k` individuals from the population and returns the one
        with the highest fitness.

        Parameters
        ----------
        population : list of Individual
            The current population of individuals.
        k : int, optional
            Number of individuals in the tournament bracket. Default is 5.

        Returns
        -------
        Individual
            The fittest individual from the tournament.
        """
        if k > len(population):
            raise ValueError(
                f"Cannot run tournament: k={k} is greater than population size={len(population)}."
            )

        bracket = random.sample(population, k=k)
        fitnesses = [(i, self.evaluate(i)) for i in bracket]
        sorted_bracket = list(sorted(fitnesses, key=lambda x: x[1], reverse=True))

        return sorted_bracket[0][0]

    def run(self, generations: int = 50):
        """"""
        # stats
        best_individual: Individual = None
        best_fitness = 0
        history = []

        population = self._generate_population()

        for generation in range(generations):

            fitnesses = [(i, self.evaluate(i)) for i in population]
            sorted_pop = list(sorted(fitnesses, key=lambda x: x[1], reverse=True))

            # update stats
            best_pop_ind, best_pop_fitness = sorted_pop[0]
            if best_pop_fitness > best_fitness:
                best_fitness = best_pop_fitness
                best_individual = best_pop_ind

            avg_pop_fitness = sum(x[1] for x in fitnesses) / len(fitnesses)

            history.append((generation, best_fitness, avg_pop_fitness))

            # create next generation
            next_generation = []

            # add elites
            n_elite = max(1, int(self.elitism_rate * self.population_size))
            elite = [x[0] for x in sorted_pop][:n_elite]
            next_generation.extend(elite)

            while len(next_generation) < self.population_size:
                parent1 = self.tournament(population)
                parent2 = self.tournament(population)

                if random.random() < self.crossover_rate:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # always add one child, if theres room add another
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)

            assert (
                len(next_generation) == self.population_size
            ), f"{len(next_generation)} != {self.population_size}"

            population = next_generation

        return best_individual, best_fitness, history
