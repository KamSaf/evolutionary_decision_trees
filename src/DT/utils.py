from random import shuffle
import math
from typing import Iterable, List, Dict
from config import DECISION_COLUMN_SYMBOL, OUTPUT_PATH


def randomize_data(path: str, output_path: str) -> None:
    """
    Function for randomizing data file and saving it to new file.

    Parameters:
        path (str): path to dataset file
        output_path (str): path to file where randomized data is to be saved
    """
    with open(path, "r") as read_file:
        file = [line for line in read_file]
    shuffle(file)
    with open(output_path, "w") as save_file:
        for line in file:
            save_file.write(line)


def read_data(
    path: str, sep: str = ",", drop_col: List[int] = [0], dec_attr_id: int = 1
) -> Dict[int, Dict[str, str | float]]:
    """
    Function for reading data from a file without headers
    (.csv is default format) and creating new headers.

    Parameters:
        path (str): path to dataset file
        sep (str): separator (between columns) used in data file
        drop_col (List[int]): indexes of columns to be ignored
        dec_attr_id (int): id of decision column

    Returns:
        Dict[int, Dict[str, str | float]]: key - row id, value - dictionary with column headers and its values
    """
    data = {}
    with open(path, "r") as file:
        for line in file:
            load_line(
                data,
                line=line.strip().split(sep),
                drop_col=drop_col,
                dec_attr_id=dec_attr_id,
            )
    return data


def load_line(
    data: Dict[int, Dict[str, str | float]],
    line: list[str],
    drop_col: List[int] = [0],
    dec_attr_id: int = 1,
) -> None:
    """
    Function for loading row of data into data dictionary.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary
        line (list[str]): data row as list of strings
        drop_col (List[int]): indexes of columns to be ignored
        dec_attr_id (int): id of decision column

    """
    row_id = len(data.keys())
    data[row_id] = {}
    col_id = 0
    for i, el in enumerate(line):
        if i in drop_col:
            continue
        if i == dec_attr_id:
            attr = DECISION_COLUMN_SYMBOL
        else:
            col_id += 1
            attr = f"c{col_id}"
            el = float(el)
        data[row_id][attr] = el


def get_attr_names(data: Dict[int, Dict[str, str | float]]) -> List[str]:
    """
    Function for returning names of all attributes in dataset.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary

    Returns:
        attr_names (list[str]): lisit of dataset attributes
    """
    return list(data[0].keys())


def get_unique_values(
    data: Dict[int, Dict[str, str | float]],
) -> Dict[str, List[str | float]]:
    """
    Function for returning unique values of attributes.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary

    Returns:
        unique_attr_vals (Dict[str, List[str | float]): key - attribute name, value - list of unique values in column
    """
    unique_values = {}
    for record in data.values():
        for attr, value in record.items():
            if attr not in unique_values.keys():
                unique_values[attr] = [value]
            elif value not in unique_values[attr]:
                unique_values[attr].append(value)
    return {attr: sorted(values) for attr, values in unique_values.items()}


def get_attr_vals(
    data: Dict[int, Dict[str, str | float]], attr: str
) -> List[str | float]:
    """
    Returns all values from a column.

    Parameters:
        data (Dict[int, dict[str, str | float]]): dataset as dictionary
        attr (str): attribute (column) name

    Returns:
        attr_vals (List[str | float]) - list of all attribute (column) values
    """
    return [record[attr] for record in data.values()]


def get_value_count(
    data: Dict[int, dict[str, str | float]],
) -> Dict[str, Dict[str | float, int]]:
    """
    Function for returning number of appearances of value in column.

    Parameters:
        data (Dict[int, dict[str, str | float]]): dataset as dictionary

    Returns:
        attr_val_count (dict[str, int]) - key - attribute name, value - dictionary with attribute value and number of appearances
    """
    unique_vals = get_unique_values(data)
    attr_val_count = {}
    for attr, values in unique_vals.items():
        val_app_num = {val: get_attr_vals(data, attr).count(val) for val in values}
        attr_val_count[attr] = val_app_num
    return attr_val_count
