from pathlib import Path
from random import seed

ROOT = Path(__file__).parent.parent
DECISION_COLUMN_SYMBOL = "d"
DATA_FILE_PATH = f"{ROOT}/data/wdbc.data"
INDENT = "      "
SEED = 123

seed(SEED)
