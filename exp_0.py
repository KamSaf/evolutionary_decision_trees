from random import seed
from src.GP.gp_algorithm import GP
from src.DT.node import Node
from src.DT.utils import randomize_data, read_data, evaluate

seed(42)

# WDBC dataset

randomize_data("data/wdbc_long.data", "random_data.data")
data = read_data("random_data.data")

gp = GP(
    data,
)
best_trees = gp.run()
last_tree = best_trees[-1][0]

tree = Node.build_tree_struct(data=gp.train_ds)
tree_benchmark = evaluate(tree.test_tree(gp.test_ds))

print(
    "\nC4.5 tree result: ",
    tree_benchmark,
    ", depth: ",
    tree.get_depth(),
    " nodes: ",
    tree.get_nodes_count(),
)
print(
    "GP algorithm tree: ",
    evaluate(last_tree.test_tree(gp.test_ds)),
    ", depth: ",
    last_tree.get_depth(),
    ", nodes: ",
    last_tree.get_nodes_count(),
)

# CAR_SIL dataset

randomize_data("data/car_sil.data", "random_data.data")
data = read_data(
    "random_data.data", sep=" ", drop_col=[], dec_attr_id=18
)  # num_of_attrs=8, max_tree_depth=10


# gp = GP(
#     data,
#     population_size=100,
#     elite_num=5,
#     crossover_rate=0.7,
#     mutation_rate=0.2,
#     generations=50,
#     num_of_attrs=10,
#     train_ds_ratio=0.6,
#     valid_ds_ratio=0.25,
#     test_ds_ratio=0.15,
#     max_tree_depth=8,
# )
gp = GP(
    data,
)
best_trees = gp.run()
last_tree = best_trees[-1][0]

tree = Node.build_tree_struct(data=gp.train_ds)
tree_benchmark = evaluate(tree.test_tree(gp.test_ds))

print(
    "\nC4.5 tree result: ",
    tree_benchmark,
    ", depth: ",
    tree.get_depth(),
    " nodes: ",
    tree.get_nodes_count(),
)
print(
    "GP algorithm tree: ",
    evaluate(last_tree.test_tree(gp.test_ds)),
    ", depth: ",
    last_tree.get_depth(),
    ", nodes: ",
    last_tree.get_nodes_count(),
)

# IONOSPHERE dataset

randomize_data("data/ionosphere.data", "random_data.data")
data = read_data("random_data.data", sep=",", drop_col=[0], dec_attr_id=34)

gp = GP(data)
best_trees = gp.run()
last_tree = best_trees[-1][0]

tree = Node.build_tree_struct(data=gp.train_ds)
tree_benchmark = evaluate(tree.test_tree(gp.test_ds))

print(
    "\nC4.5 tree result: ",
    tree_benchmark,
    ", depth: ",
    tree.get_depth(),
    " nodes: ",
    tree.get_nodes_count(),
)
print(
    "GP algorithm tree: ",
    evaluate(last_tree.test_tree(gp.test_ds)),
    ", depth: ",
    last_tree.get_depth(),
    ", nodes: ",
    last_tree.get_nodes_count(),
)
