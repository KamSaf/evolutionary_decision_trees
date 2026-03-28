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
start = time.time()
tree_1 = Node.build_tree_struct(data=train, max_tree_depth=3)
stop = time.time()
stats = tree_1.test_tree(test)
results_1 = evaluate(stats)
print("Build time: ", stop - start)
print(results_1)
print("\nRANDOM TREE")
start = time.time()
tree_2 = Node.build_tree_struct(data=train, random=True, max_tree_depth=3)
stop = time.time()
stats = tree_2.test_tree(test)
results_2 = evaluate(stats)
print("Build time: ", stop - start)
print(results_2)
print(tree_1)
print()
print(tree_2)
print(tree_1.get_depth())
print(tree_2.get_depth())
