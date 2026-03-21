from pathlib import Path

ROOT = Path(__file__).parent.parent
DECISION_COLUMN_SYMBOL = "d"
DATA_FILE_PATH = f"{ROOT}/data/wdbc.data"
OUTPUT_PATH = f"{ROOT}/tree.txt"
INDENT = "      "
TEST_DATA_RATIO = 0.3
