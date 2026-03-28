from src.GP.gp_algorithm import GP
from src.DT.utils import (
    randomize_data,
    read_data,
    evaluate,
)

randomize_data("data/wdbc_long.data", "random_data.data")
data = read_data("random_data.data")


gp = GP(data)

gp._GP__init_population(log=True)

print(len(gp.population))
for tree in gp.population:
    print(evaluate(tree.test_tree(gp.test_ds)))
