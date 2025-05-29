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
