from pathlib import Path
from random import seed

ROOT = Path(__file__).parent.parent
DECISION_COLUMN_SYMBOL = "d"
DATA_FILE_PATH = f"{ROOT}/data/wdbc.data"
OUTPUT_PATH = f"{ROOT}/tree.txt"
INDENT = "      "
SEED = 123

seed(SEED)
