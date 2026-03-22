from src.DT.node import Node
from src.DT.utils import randomize_data, read_data


randomize_data("data/wdbc.data", "random_data.data")
data = read_data("random_data.data")
root = Node()
Node.build_tree_struct(data=data)
