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
        """
        Creates population of decision trees
        """
        # TODO
        pass

    def __evaluate_population(self) -> List[Tuple[Node, float]]:
        """
        Calculates fitness for each decision tree in population.

        Returns:
            List[Tuple[Node, float]]: list of trees with their fitness values
        """
        # TODO
        return [(Node(), 0.1)]

    def __roulette_selection(self) -> Node:
        """
        Selects tree from population using roulette wheel method.

        Returns:
            Node: decision tree selected from population
        """
        # TODO
        return Node()

    def __tournament_selection(self) -> Node:
        """
        Selects tree from population using tournament method.

        Returns:
            Node: decision tree selected from population
        """
        # TODO
        return Node()

    def __elitism(self) -> List[Node]:
        """
        Selects best fitted trees in population.

        Returns:
            List[Node]: list of best trees in population
        """
        # TODO
        return []

    @staticmethod
    def __crossover(
        parents: Tuple[Node, Node], crossover_rate: float
    ) -> Tuple[Node, Node]:
        """
        Implementation of crossover genetic operator. Switches randomly
        chosen subtrees between two decision trees.

        Parameters:
            parents (Tuple[Node, Node]): parents, of which subtrees are to be replaced
            crossover_rate (float): probability of crossover taking place

        Returns:
            Tuple[Node, Node]: pair of offsprings
        """
        # TODO
        return parents

    @staticmethod
    def __mutation(tree: Node, mutation_rate: float) -> Node:
        """
        Implementation of mutation genetic operator. Replaces attribute
        and value threshold in randomly chosen node of decision tree.

        Parameters:
            tree (Node): decision tree to be mutated
            mutation_rate (float): probability of mutation taking place

        Returns:
            Node: mutated (or not) tree
        """
        # TODO
        return tree

    def run(self) -> Node:
        """
        Runs genetic programming algorithm on population of decision trees.

        Returns:
            Node: best of the evolved decision trees
        """
        # TODO
        return Node()
