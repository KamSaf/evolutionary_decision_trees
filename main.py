from random import seed
from src.GP.gp_algorithm import GP
from src.DT.node import Node
from src.DT.utils import randomize_data, read_data, evaluate

seed(42)
randomize_data("data/wdbc_long.data", "random_data.data")
data = read_data("random_data.data")

gp = GP(data, population_size=10, elite_num=3, crossover_rate=0.7, mutation_rate=0.2)

gp._GP__init_population(display_logs=True)  # type: ignore
best_trees = gp.run(display_logs=True)

print([el[1] for el in best_trees])
last_tree = best_trees[-1][0]

tree = Node.build_tree_struct(data=gp.train_ds)
tree_benchmark = evaluate(tree.test_tree(gp.train_ds))

print("\nC4.5 result: ", tree_benchmark)
print("GP algorithm: ", evaluate(last_tree.test_tree(gp.test_ds)))
