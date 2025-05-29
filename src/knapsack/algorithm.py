from dataclasses import dataclass


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
