from typing import List, Dict, Tuple

# from copy import deepcopy
from uuid import UUID
from random import choice, uniform
from enum import Enum
from src.DT.node import Node
from src.DT.utils import (
    create_datasets,
    get_attr_names,
    get_col,
    DECISION_COLUMN_SYMBOL,
)


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
        self.dataset = dataset
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
        # TODO DEEPCOPY!!!!!!!
        return Node()

    def __tournament_selection(self) -> Node:
        """
        Selects tree from population using tournament method.

        Returns:
            Node: decision tree selected from population
        """
        # TODO DEEPCOPY!!!!!!!
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
    def __crossover(parents: List[Node]) -> List[Node]:
        """
        Implementation of crossover genetic operator. Switches randomly
        chosen subtrees between two decision trees.

        Parameters:
            parents (List[Node]): parents, of which subtrees are to be replaced

        Returns:
            Tuple[Node, Node]: pair of offsprings
        """
        subtrees_ids = [p.get_random_node_id() for p in parents]
        queue_subtree_1 = [node for node in parents[0].children]
        for n1 in queue_subtree_1:
            if n1.id != subtrees_ids[0]:
                queue_subtree_1 += n1.children
                continue
            queue_subtree_2 = [node for node in parents[1].children]
            for n2 in queue_subtree_2:
                if n2 != subtrees_ids[1]:
                    queue_subtree_2 += n2.children
                    continue
                n1.switch_nodes(n2)
        return parents

    @staticmethod
    def __mutate(
        tree: Node,
        id: UUID,
        train_ds: Dict[int, Dict[str, str | float]],
    ) -> None:
        """
        Implementation of mutation genetic operator. Replaces attribute
        and value threshold in randomly chosen node of decision tree.

        Parameters:
            tree (Node): decision tree to be mutated
            id (UUID): id of trees node to mutate
            train_ds (Dict[int, Dict[str, str | float]]): dataset as dictionary
        """
        for node in tree.children:
            if node.id == id:
                attrs = get_attr_names(train_ds)
                attrs.remove(DECISION_COLUMN_SYMBOL)
                new_attr = choice(attrs)
                attr_vals = get_col(train_ds, new_attr)
                min_thresh, max_tresh = min(attr_vals), max(attr_vals)
                node.update_node_attr(
                    new_attr, uniform(min_thresh, max_tresh)  # type: ignore
                )
            else:
                GP.__mutate(node, id, train_ds)

    def run(self) -> List[Tuple[Node, float]]:
        """
        Runs genetic programming algorithm on population of decision trees.

        Returns:
            List[Tuple[Node, float]]: list of best trees (and their fitness) from each generation
        """
        self.__init_population()
        best_trees = []
        for i in range(self.generations):
            best_trees.append(self.__evaluate_population()[-1])
            new_population = self.__elitism()
            while len(new_population) < self.population_size:
                parents = [
                    (
                        self.__tournament_selection()
                        if self.selection_method == "tournament"
                        else self.__roulette_selection()
                    )
                    for _ in range(2)
                ]
                offspring = (
                    GP.__crossover(parents)
                    if uniform(0, 1) <= self.crossover_rate
                    else parents
                )

                for o in offspring:
                    if uniform(0, 1) <= self.mutation_rate:
                        id = o.get_random_node_id()
                        GP.__mutate(o, id, self.train_ds)
                    new_population.append(o)
        return best_trees
