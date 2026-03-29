from enum import Enum
from typing import Dict
from datetime import datetime
from src.DT.node import Node
from src.DT.utils import evaluate


class SelectionMethods(Enum):
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette"


class FitnessMetric(Enum):
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"


def log(msg: str) -> None:
    """
    Logs given message with datetime in console.

    Paramters:
        str: message to be displayed
    """
    print(datetime.now().strftime(f"[%H:%M:%S] {msg}"))


def get_fitness(
    tree: Node,
    valid_ds: Dict[int, Dict[str, str | float]],
    fitness_metric: FitnessMetric,
    tree_depth_modifier: float,
) -> float:
    """
    Returns tree fitness calculated with F1 score or accuracy and tree depth modifier
    multiplied by given decision tree depth.

    Parameters:
        tree (Node): decison tree from population
        valid_ds (Dict[int, Dict[str, str | float]]): validation dataset as dictionary
        fitness_metric ("accuracy" or "f1_score"): classification quality metric used to calculate fitness
        tree_depth_modifier (float): modifier to calculate tree depth penalty

    Returns:
        float: decision tree fitness
    """
    accuracy, recall, precision = evaluate(tree.test_tree(valid_ds))
    f1_score = (2 * precision * recall) / (precision + recall)
    metric_val = accuracy if fitness_metric == "accuracy" else f1_score
    depth = tree.get_depth()
    return metric_val - depth * tree_depth_modifier
