from typing import List, Dict, Tuple
from enum import Enum
from src.DT.node import Node
from src.DT.utils import create_datasets


class SelectionMethods(Enum):
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette"


class GP:
    def __init__(
        self,
        dataset: Dict[int, Dict[str, str | float]],
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.1,
        selection_method: SelectionMethods = SelectionMethods.TOURNAMENT,
        tournament_size: int = 5,
        max_tree_depth: int = 6,
        random_pop_ratio: float = 0.75,
        train_ds_ratio: float = 0.7,
        valid_ds_ratio: float = 0.15,
        test_ds_ratio: float = 0.15,
        elite_individuals_num: int = 5,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.max_tree_depth = max_tree_depth
        self.random_pop_ratio = random_pop_ratio
        self.train_ds, self.valid_ds, self.test_ds = create_datasets(
            dataset,
            train_ratio=train_ds_ratio,
            valid_ratio=valid_ds_ratio,
            test_ratio=test_ds_ratio,
        )
        self.elite_individuals_num = elite_individuals_num
        self.population: List[Node] = []

    def __init_population(self) -> None:
        # TODO
        pass

    def __evaluate_population(self) -> List[Tuple[Node, float]]:
        # TODO
        return [(Node(), 0.1)]

    def __roulette_selection(self) -> Node:
        # TODO
        return Node()

    def __tournament_selection(self) -> Node:
        # TODO
        return Node()

    def __elitism(self) -> List[Node]:
        # TODO
        return []

    @staticmethod
    def __crossover(
        parents: Tuple[Node, Node], crossover_rate: float
    ) -> Tuple[Node, Node]:
        # TODO
        return parents

    @staticmethod
    def __mutation(tree: Node, mutation_rate: float) -> Node:
        # TODO
        return tree

    def run(self) -> Node:
        # TODO
        return Node()
