import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from scipy.stats import norm
from multiprocessing import cpu_count
import os


def compute_rolling_intersection_pct(arr1, arr2):
    """Compute the rolling intersection percentage between two arrays"""
    if len(arr1) != len(arr2):
        raise ValueError('Arrays must be of the same length')

    arr_length = len(arr1)
    overlap_rolling_pct = np.zeros(arr_length - 1)

    for i in range(arr_length - 1):
        elems_in_arr1 = arr1[: i + 1]
        elems_in_arr2 = arr2[: i + 1]

        overlap_count = np.intersect1d(elems_in_arr1, elems_in_arr2)

        overlap_rolling_pct[i] = len(overlap_count) / (i + 1)

    return overlap_rolling_pct


def compute_num_cores_to_utilise():
    num_cores = cpu_count()

    if num_cores > 8:
        return 8
    elif num_cores > 2:
        return num_cores - 2
    else:
        return 1


def compute_confidence_interval(error, confidence_level=0.95):
    """
    Computes a confidence interval for a given error. By default, this computes the 95% confidence interval.
    """
    return norm.ppf((1 + confidence_level) / 2) * error


def save_dataframe_to_csv(df: pd.DataFrame, script_filepath: str, name: str, **kwargs):
    cwd = os.path.dirname(os.path.realpath(script_filepath))
    create_output_dir_if_required(script_filepath)
    df.to_csv(os.path.join(cwd, f'../outputs/{name}'), **kwargs)


def load_dataframe_from_csv(
    df: pd.DataFrame, script_filepath: str, name: str, **kwargs
):
    cwd = os.path.dirname(os.path.realpath(script_filepath))
    return pd.read_csv(os.path.join(cwd, f'../outputs/{name}'), **kwargs)


def save_dict_to_json(data: dict, script_filepath: str, name: str):
    cwd = os.path.dirname(os.path.realpath(script_filepath))

    create_output_dir_if_required(script_filepath)

    with open(os.path.join(cwd, f'../outputs/{name}'), 'w') as f:
        json.dump(data, f, indent=4)


def load_dict_from_json(script_filepath: str, name: str):
    cwd = os.path.dirname(os.path.realpath(script_filepath))

    with open(os.path.join(cwd, f'../outputs/{name}'), 'r') as f:
        return json.load(f)


def create_output_dir_if_required(script_filepath: str):
    cwd = os.path.dirname(os.path.realpath(script_filepath))

    if not os.path.exists(os.path.join(cwd, '../outputs')):
        os.makedirs(os.path.join(cwd, '../outputs'))


def save_fig(script_filepath: str, name: str, **kwargs):
    create_output_dir_if_required(script_filepath)

    cwd = os.path.dirname(os.path.realpath(script_filepath))
    plt.savefig(os.path.join(cwd, f'../outputs/{name}'), bbox_inches='tight', **kwargs)


def load_dataset(
    dataset: str, drop_columns: list[str] = None, standardise=False
) -> pd.DataFrame:
    cwd = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(os.path.join(cwd, f'../datasets/{dataset}'))

    if drop_columns:
        data = data.drop(drop_columns, axis=1)

    if standardise:
        if 'classification' in data.columns:
            # Don't standardise the classification column if it is still in the dataset
            data = pd.DataFrame(
                np.column_stack(
                    [
                        StandardScaler().fit_transform(data.values[:, 0:-1]),
                        data.values[:, -1],
                    ]
                ),
                columns=data.columns,
            )
        else:
            data = pd.DataFrame(
                StandardScaler().fit_transform(data.values), columns=data.columns
            )

    return data
