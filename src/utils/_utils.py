import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from scipy.stats import norm


def compute_confidence_interval(error, confidence_level=0.95):
    """
    Computes a confidence interval for a given error. By default, this computes the 95% confidence interval.
    """
    return norm.ppf((1 + confidence_level) / 2) * error


def save_dict_to_json(dict: dict, script_filepath: str, name: str):
    cwd = os.path.dirname(os.path.realpath(script_filepath))

    if not os.path.exists(os.path.join(cwd, '../outputs')):
        os.makedirs(os.path.join(cwd, '../outputs'))

    with open(os.path.join(cwd, f'../outputs/{name}'), 'w') as f:
        json.dump(dict, f, indent=4)


def load_dict_from_json(script_filepath: str, name: str):
    cwd = os.path.dirname(os.path.realpath(script_filepath))

    if not os.path.exists(os.path.join(cwd, '../outputs')):
        os.makedirs(os.path.join(cwd, '../outputs'))

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
