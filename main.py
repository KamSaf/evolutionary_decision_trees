from random import seed
from src.GP.gp_algorithm import GP
from src.DT.node import Node
from src.DT.utils import randomize_data, read_data, evaluate

seed(42)
# randomize_data("data/wdbc_long.data", "random_data.data")
# data = read_data("random_data.data")

randomize_data("data/car_sil.data", "random_data.data")
data = read_data(
    "random_data.data", sep=" ", drop_col=[], dec_attr_id=18
)  # num_of_attrs=8, max_tree_depth=10

# randomize_data("data/leaf.csv", "random_data.data")
# data = read_data("random_data.data", drop_col=[], dec_attr_id=0)

# randomize_data("data/sonar.all-data", "random_data.data")
# data = read_data("random_data.data", drop_col=[], dec_attr_id=60)


# gp = GP(
#     data,
#     population_size=100,
#     elite_num=3,
#     crossover_rate=0.7,
#     mutation_rate=0.2,
# )


for gen in [5, 10, 20, 30, 40, 50]:
    # kiedy zmniejszymi zbiór treningowy i zwiększymy walidacyjny to przy 10 generacjach mamy
    # bardzo kompaktowe i płytkie drzewo o jakości porównywalnej z tradycyjnym c4.5 na większym zbiorze testowym

    # przy wzroście liczby generacji jest jescze lepiej
    gp = GP(
        data,
        population_size=100,
        elite_num=3,
        crossover_rate=0.7,
        mutation_rate=0.2,
        generations=gen,
        num_of_attrs=10,
        train_ds_ratio=0.6,
        valid_ds_ratio=0.25,
        test_ds_ratio=0.15,
    )
    best_trees = gp.run()
    last_tree = best_trees[-1][0]

    tree = Node.build_tree_struct(data=gp.train_ds, max_tree_depth=10)
    tree_benchmark = evaluate(tree.test_tree(gp.test_ds))

    random_tree = Node.build_tree_struct(
        data=gp.train_ds, random=True, max_tree_depth=10
    )
    random_tree_benchmark = evaluate(random_tree.test_tree(gp.test_ds))

    print(
        "\nC4.5 tree result: ",
        tree_benchmark,
        ", depth: ",
        tree.get_depth(),
        " nodes: ",
        tree.get_nodes_count(),
    )
    print(
        "Random tree result: ",
        random_tree_benchmark,
        ", depth: ",
        random_tree.get_depth(),
        " nodes: ",
        random_tree.get_nodes_count(),
    )
    print(
        "GP algorithm tree: ",
        evaluate(last_tree.test_tree(gp.test_ds)),
        ", depth: ",
        last_tree.get_depth(),
        ", nodes: ",
        last_tree.get_nodes_count(),
    )
