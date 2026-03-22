from random import shuffle
import math
from typing import List, Dict, Tuple
from src.DT.config import DECISION_COLUMN_SYMBOL

# import time


def randomize_data(path: str, output_path: str) -> None:
    """
    Randomizes data file and saving it to new file.

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
    Reads data from a file without headers
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
    Loads row of data into data dictionary.

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
    Returns names of all attributes in dataset.

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
    Returns unique values of attributes.

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


def get_col(data: Dict[int, Dict[str, str | float]], attr: str) -> List[str | float]:
    """
    Returns all values from a column.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary
        attr (str): attribute (column) name

    Returns:
        col_vals (List[str | float]) - list of all attribute (column) values
    """
    return [record[attr] for record in data.values()]


def get_dataset_length(data: Dict[int, Dict[str, str | float]]) -> int:
    """
    Returns length of dataset.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary
    Returns:
        dataset_length (int) - length of dataset (decision column length)
    """
    return len(get_col(data, DECISION_COLUMN_SYMBOL))


def get_value_count(
    data: Dict[int, Dict[str, str | float]],
) -> Dict[str, Dict[str | float, int]]:
    """
    Returns number of appearances of value in column.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary

    Returns:
        attr_val_count (Dict[str, Dict[str | float, int]]) - key - attribute name, value - dictionary with
        attribute value and number of appearances
    """
    unique_vals = get_unique_values(data)
    attr_val_count = {}
    for attr, values in unique_vals.items():
        val_app_num = {val: get_col(data, attr).count(val) for val in values}
        attr_val_count[attr] = val_app_num
    return attr_val_count


def get_dominant_attr_val(
    data: Dict[int, Dict[str, str | float]], attr: str
) -> str | float:
    """
    Returns dominant value in given dataset column

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary

    Returns:
        dom_val (str | float) - dominant value in column
    """
    values_count = get_value_count(data)[attr]
    max_val = ""
    max_num = 0
    for val, num in values_count.items():
        if num > max_num:
            max_val = val
            max_num = num
    return max_val


def get_values_probabilities(
    data: Dict[int, Dict[str, str | float]],
) -> Dict[str, Dict[str | float, float]]:
    """
    Returns probabilities of every attribute value.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary

    Returns:
        values_propabilities (Dict[str, Dict[str | float, float]]): key - attribute name, value - dictionary with unique
        values as keys and its propabilities as values
    """
    probabilities = {}
    values_count = get_value_count(data)
    for attr, values in values_count.items():
        col_length = len(get_col(data, attr))
        probabilities[attr] = {
            val: float(num) / float(col_length) for val, num in values.items()
        }
    return probabilities


def get_entropy(probabilities: List[float]) -> float:
    """
    Calculates entropy from a list of propabilities.

    Parameters:
        probabilities (List[float]): list of attribute values probabilities

    Return:
        entropy (float): calculated entropy
    """
    filtered_propabilities = (p for p in probabilities if p != 0)
    return -1 * sum([p * math.log2(p) for p in filtered_propabilities])


def split_dataset(
    data: Dict[int, Dict[str, str | float]], thresh: float, attr: str
) -> List[Dict[int, Dict[str, str | float]]]:
    """
    Splits dataset by attribute value threshold.

    Parameters:
        data (ddict[str, list[float] | list[str]]): dataset as dictionary
        thresh (float): attribute values threshold
        attr (str): attribute name (column header)

    Returns:
        split_data (List[Dict[int, Dict[str, str | float]]]): list of two data subsets split by given threshold
    """
    subsets = [{}, {}]
    if attr == DECISION_COLUMN_SYMBOL:
        return subsets
    for record in data.values():
        if record[attr] <= thresh:  # type: ignore
            subsets[0][len(subsets[0])] = record
        else:
            subsets[1][len(subsets[1])] = record
    return subsets


def get_column_entropy(
    data: Dict[int, Dict[str, str | float]], attr: str = DECISION_COLUMN_SYMBOL
) -> float:
    """
    Calculates entropy for a given column in dataset.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary
        attr (str): attribute name

    Returns:
        entropy (float): calculated entorpy of a given column
    """
    # TODO WYLICZANIE PRAWDOPODOBIEŃSTWA TYLKO DLA KOLUMNY !!!!
    values_propabilities = list(get_values_probabilities(data)[attr].values())
    return get_entropy(values_propabilities)


def get_info(
    data: Dict[int, Dict[str, str | float]], thresh: float, attr: str
) -> float:
    """
    Calculates info for a given attribute in dataset dictionary.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary
        threshold (float): attribute values threshold
        attr (str): split attribute name

    Returns:
        attr_info (float): calculated info of given attribute
    """
    data_subsets = split_dataset(data, thresh, attr=attr)
    info = 0
    for subset in data_subsets:
        entropy = get_column_entropy(subset)
        info += (get_dataset_length(subset) / get_dataset_length(data)) * entropy
    return info


def get_gain_ratio(
    data: Dict[int, Dict[str, str | float]],
    attr: str,
    split_points: List[float],
) -> Tuple[float, float] | Tuple[None, None]:
    """
    Returns tuple containing best threshold and its gain ratio.

    Parameters:
        data (Dict[int, Dict[str, str | float]]): dataset as dictionary
        attr (str): name of attribute
        split_points (List[float]): list of value splitting points

    Returns:
        gain_ratios (Tuple[float, float] | Tuple[None, None]):
    """
    gain_ratios = {}
    if len(split_points) < 1:
        return (None, None)
    decision_column_entropy = get_column_entropy(data)
    for thresh in split_points:
        # start = time.time()
        info = get_info(data, thresh, attr)
        # stop = time.time()
        gain_ratios[thresh] = (decision_column_entropy - info) / decision_column_entropy
    max_gain_ratio = -1.0
    max_thresh = 0.0
    for thresh, gain_ratio in gain_ratios.items():
        if gain_ratio > max_gain_ratio:
            max_gain_ratio = gain_ratio
            max_thresh = thresh
    return max_thresh, max_gain_ratio


def get_max_ratio_attr(
    data: Dict[int, Dict[str, str | float]],
) -> Tuple[str, float, float]:
    """
    Returns attribute name and threshold with highest gain ratio in given dataset.

    Parameters:
        data (dict[str, list[float] | list[str]]): dataset as dictionary

    Returns:
        (tuple[str, float, float, float]): attribute name, gain ratio, threshold
    """
    attrs = get_attr_names(data)
    attrs.remove(DECISION_COLUMN_SYMBOL)
    ratios = {}
    for attr in attrs:
        sorted_vals = sorted(get_col(data, attr))
        split_points = [
            (sorted_vals[i] + sorted_vals[i + 1]) / 2  # type: ignore
            for i in range(len(sorted_vals) - 1)
        ]
        thresh, gain_ratio = get_gain_ratio(data, attr, split_points)  # type: ignore
        ratios[attr] = (thresh, gain_ratio)
    max_gain_ratio = -1.0
    max_thresh = 0.0
    max_attr = ""
    for attr, (thresh, gain_ratio) in ratios.items():
        if gain_ratio > max_gain_ratio:
            max_gain_ratio = gain_ratio
            max_thresh = thresh
            max_attr = attr
    return max_attr, max_gain_ratio, max_thresh
