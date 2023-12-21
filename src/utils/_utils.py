import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib.axes import Axes
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json


def identify_most_discriminative_features(
    X: pd.DataFrame, loading_pct_threshold=0.01
) -> pd.DataFrame:
    """Compute the most discriminative features using PCA.

    Parameters
    ----------
    X
        2D DataFrame of data
    loading_pct_threshold
        The threshold for the percentage of variance explained by a loading for it to be considered a discriminative
        feature

    Returns
    -------
    DataFrame
        A DataFrame of the most discriminative features, with the loadings of each feature on the first two principal
        components, and the index being the feature names.
    """
    X = X.copy()
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    pca = PCA(n_components=2)
    pca.fit(X)
    z = pca.transform(X)

    loadings = pca.components_.T
    loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=X.columns)

    highest_loadings = loadings[
        (loadings['PC1'] ** 2 > loading_pct_threshold)
        | (loadings['PC2'] ** 2 > loading_pct_threshold)
    ]

    return highest_loadings


def identify_outliers(X: pd.DataFrame | np.ndarray, z_threshold=3.0):
    """
    Identify outliers in a DataFrame by computing the z-scores of each value in the DataFrame, and then identifying
    values that are more than z_threshold standard deviations away from the mean.

    Parameters
    ----------
    X
        2D DataFrame of data
    z_threshold
        The threshold for the z-score of a value to be considered an outlier

    Returns
    -------
    NDArray
        A 2D numpy array of booleans indicating whether a value is an outlier. This array has the same shape as X.
    """
    X = X.copy()
    z_scores = StandardScaler().fit_transform(X)

    # A 2d numpy array of booleans indicating whether a value is an outlier
    outliers = (np.abs(z_scores) > z_threshold) & (
        np.broadcast_to(X.var(axis=0), z_scores.shape) > 1e-8
    )

    return outliers


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


def format_axes(ax: Axes, **kwargs):
    if ax.get_legend():
        ax.legend(
            facecolor='white',
            loc='best' if 'legend_loc' not in kwargs.keys() else kwargs['legend_loc'],
        )

    # Make the axes the plots have a white background
    ax.set_facecolor('white')

    # Format the spines
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_edgecolor('k')
        ax.spines[side].set_linewidth(0.5)

    # Add minor ticks to the axes
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Turn on all ticks
    ax.tick_params(
        which='both',
        top=True if 'ticks_top' not in kwargs.keys() else kwargs['ticks_top'],
        bottom=True if 'ticks_bottom' not in kwargs.keys() else kwargs['ticks_bottom'],
        left=True if 'ticks_left' not in kwargs.keys() else kwargs['ticks_left'],
        right=True if 'ticks_right' not in kwargs.keys() else kwargs['ticks_right'],
    )

    ax.tick_params(which='minor', length=2, color='k', direction='out')
    ax.tick_params(which='major', length=4, color='k', direction='out')
