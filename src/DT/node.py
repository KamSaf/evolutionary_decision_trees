from typing import Dict, List
from random import choice
from uuid import uuid1, UUID
from src.DT.config import (
    DECISION_COLUMN_SYMBOL,
    DATA_FILE_PATH,
    INDENT,
)
from src.DT.utils import (
    read_data,
    get_max_ratio_attr,
    get_unique_values,
    split_dataset,
    get_dominant_attr_val,
    get_random_ratio_attr,
)


class Node:
    def __assign_parent(self) -> None:
        """
        Recursively assigns parent identificator
        to nodes children.
        """
        if len(self.children) == 0:
            return
        for c in self.children:
            c.parent_id = self.id
            c.__assign_parent()

    def __init__(
        self,
        label: str = "node",
        children: list["Node"] | None = None,
        val: str = "None",
        parent_id: UUID | None = None,
    ):
        self.id = uuid1()
        self.label = label
        self.val = val
        self.parent_id = parent_id
        self.children = [] if children is None else children
        self.__assign_parent()

    def __get_child_by_value(self, val: str) -> "Node | None":
        """
        Returns child of a node by value.

        Parameters:
            val (str): value of a node to look for

        Returns:
            (Node | None): children with given value
        """
        target = [c for c in self.children if c.val == val]
        return target[0] if len(target) else None

    def __append_child(self, child: "Node") -> None:
        """
        Adds node to children list.

        Parameters:
           child (Node): node to be appended
        """
        self.children.append(child)

    def get_depth(self, first_step: bool = True) -> int:
        """
        Recursively calculates depth of tree, where self is its root.

        Parameters:
            first_step (bool): flag marking first iteration (don't change)

        Returns:
            (int): depth of tree
        """
        depth = 0 if first_step else 1
        depth_of_children = (
            [c.get_depth(False) for c in self.children] if len(self.children) else []
        )
        max_children_depth = max(depth_of_children) if len(depth_of_children) > 0 else 0
        return depth + max_children_depth

    def update_node_attr(self, attr: str, thresh: float) -> None:
        """
        Updates nodes label and value with given attribute name and values threshold.

        Parameters:
            attr (str): new name of attribite
            thresh (float): new attribute value threshold
        """
        self.label = f"{attr} > {thresh}"
        for i, c in enumerate(self.children):
            c.val = f"<= {thresh}" if i == 0 else f"> {thresh}"

    def get_random_node_id(self) -> UUID:
        """
        Returns randomly chosen node id from subtree.

        Returns:
            (UUID): randomly chosen node ID
        """
        queue = self.children[:]
        nodes_ids = [self.id]
        for node in queue:
            children = node.children
            queue += children
            for c in children:
                nodes_ids.append(c.id)
        return choice(nodes_ids)

    def __to_string(self, indent: int = 0) -> str:
        """
        Recursively converts node data to string.

        Parameters:
            indent (int): indentation level (node depth)

        Returns:
            (str): node data as string
        """
        ind = INDENT * indent
        output = []
        output.append(f"\n{ind}ID: {self.id}")
        output.append(f"{ind}Label: {self.label}")
        output.append(f"{ind}Value: {self.val}")
        output.append(f"{ind}Parent: {self.parent_id if self.parent_id else None}")
        if self.children:
            output.append(f"{ind}Children:")
            for child in self.children:
                output.append(child.__to_string(indent + 1))
            output.append("\n")
        return "\n".join(output)

    def __str__(self) -> str:
        return self.__to_string()

    @staticmethod
    def build_tree_struct(
        root: "Node | None" = None,
        data: Dict[int, Dict[str, str | float]] | None = None,
        data_path: str = DATA_FILE_PATH,
        max_tree_depth: int = 8,
        split_level: int = 0,
        random: bool = False,
    ) -> "Node | None":
        """
        Builds decision tree structure.

        Parameters:
            root (Node | None): root from which tree will be built
            data (Dict[int, Dict[str, str | float]] | None): dataset as dictionary
            data_path (str): path to dataset file
            max_tree_depth (int): maximum decision tree depth
            split_level (int): current split level (do not change)
            random (bool): random tree induction flag
        Returns:
            (Node | None): decision tree
        """
        if root is None:
            root = Node()
        if "DECISION" in root.label:
            return root
        if not data:
            data = read_data(data_path)
        attr, ratio, thresh = (
            get_max_ratio_attr(data) if not random else get_random_ratio_attr(data)
        )
        split_level += 1
        if (
            abs(ratio) == 0 or split_level == max_tree_depth - 1
        ):  # may return tree consisting of one node if bad dataset is drawn
            root.label = (
                f"DECISION: {get_dominant_attr_val(data, DECISION_COLUMN_SYMBOL)}"
            )
            return root
        root.label = f"{attr} > {thresh}"
        split_data = split_dataset(data, thresh, attr)
        for i, sd in enumerate(split_data):
            decision_column_values = tuple(
                get_unique_values(sd)[DECISION_COLUMN_SYMBOL]
            )
            label = (
                f"DECISION: {decision_column_values[0]}"
                if len(decision_column_values) == 1
                else "node"
            )
            new_node = Node(
                label=label,
                val=f"<= {thresh}" if i == 0 else f"> {thresh}",
                parent_id=root.id,
            )
            root.__append_child(new_node)
            Node.build_tree_struct(new_node, sd, split_level=split_level)
        return root

    def predict(self, data_row: Dict[str, str | float]) -> str | None:
        """
        Predicts class with decision tree structure.

        Parameters:
            data_row (Dict[str, str | float]): single row from dataset

        Returns:
            (str | None): decision made with decision tree
        """
        if "DECISION" in self.label:
            return self.label
        attr, _, thresh = self.label.split(" ")
        val = data_row[attr]
        test_res = float(val) > float(thresh)
        next_step_val = f"{'>' if test_res else '<='} {thresh}"
        next_step = self.__get_child_by_value(next_step_val)
        if not next_step:
            return None
        pred = next_step.predict(data_row)
        return pred.split(" ")[1] if pred and "DECISION" in pred else pred

    def test_tree(
        self, test_ds: Dict[int, Dict[str, str | float]]
    ) -> Dict[str | float, List[int]]:
        """
        Tests decision tree classification with testing dataset.

        Parameters:
            test_ds (Dict[int, Dict[str, str | float]]): testing dataset

        Returns:
            (Dict[str, List[int]]): TP, FP, FN, TN values for each class
        """
        results = {
            dec: [0, 0, 0, 0]
            for dec in get_unique_values(test_ds)[DECISION_COLUMN_SYMBOL]
        }
        for _, row in test_ds.items():
            actual = row[DECISION_COLUMN_SYMBOL]
            pred = self.predict(row)
            for class_ in results.keys():
                if class_ == actual and pred == class_:
                    results[class_][0] += 1  # TP
                elif actual != class_ and pred == class_:
                    results[class_][1] += 1  # FP
                elif actual == class_ and pred != class_:
                    results[class_][2] += 1  # FN
                elif actual != class_ and pred != class_:
                    if class_ != pred and class_ != actual:
                        results[class_][3] += 1  # TN
        return results
