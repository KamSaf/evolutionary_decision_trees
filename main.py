from src.GP.gp_algorithm import GP
from src.DT.node import Node
from src.DT.utils import randomize_data, read_data, evaluate, create_datasets

randomize_data("data/wdbc_long.data", "random_data.data")
data = read_data("random_data.data")
# train, valid, test = create_datasets(data)
# print("Tree training")
# tree = Node.build_tree_struct(data=train, max_tree_depth=5)
# tree_res = evaluate(tree.test_tree(test))
# print("Tree: ", tree_res)
# print(tree)

# print("Random tree training")
# random_tree = Node.build_tree_struct(data=train, max_tree_depth=5, random=True)
# random_tree_res = evaluate(random_tree.test_tree(test))
# print("Random tree: ", random_tree_res)
# print(random_tree)

gp = GP(data, population_size=10)

gp._GP__init_population(display_logs=True)  # type: ignore
best_trees = gp.run(display_logs=True)

print([el[1] for el in best_trees])
last_tree = best_trees[-1][0]
print(last_tree)
print(evaluate(last_tree.test_tree(gp.test_ds)))
