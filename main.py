from src.DT.node import Node
from src.DT.utils import (
    randomize_data,
    read_data,
    create_datasets,
    evaluate,
)
import time

randomize_data("data/wdbc_long.data", "random_data.data")
data = read_data("random_data.data")
train, valid, test = create_datasets(data)

print("C4.5 TREE")
tree_1 = Node()
start = time.time()
Node.build_tree_struct(tree_1, data=train)
stop = time.time()
stats = tree_1.test_tree(test)
results_1 = evaluate(stats)
print("Build time: ", stop - start)
print(results_1)
print("\nRANDOM TREE")
tree_2 = Node()
start = time.time()
Node.build_tree_struct(tree_2, data=train, random=True)
stop = time.time()
stats = tree_2.test_tree(test)
results_2 = evaluate(stats)
print("Build time: ", stop - start)
print(results_2)
print(tree_1.get_depth())
print(tree_2.get_depth())
