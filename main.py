from src.DT.node import Node
from src.DT.utils import randomize_data, read_data
import time

randomize_data("data/wdbc_long.data", "random_data.data")
data = read_data("random_data.data")
root = Node()
start = time.time()
Node.build_tree_struct(root, data=data)
stop = time.time()
print(root)
print("Build time: ", stop - start)
