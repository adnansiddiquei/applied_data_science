import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib.axes import Axes
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def save_fig(script_filepath: str, name: str, **kwargs):
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
