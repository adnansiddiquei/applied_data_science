import numpy as np
from numpy.typing import NDArray
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator


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


def create_table(
    data: NDArray | pd.DataFrame,
    columns: list[str] = None,
    index: list[str] = None,
    figsize: tuple[int, int] = (12, 3),
) -> matplotlib.table.Table:
    if isinstance(data, np.ndarray):
        if columns is None:
            raise ValueError(
                'If the data is a numpy array, then the columns must be specified'
            )

        if index is None:
            raise ValueError(
                'If the data is a numpy array, then the index must be specified'
            )

    df = (
        pd.DataFrame(
            data,
            columns=columns,
            index=index,
        )
        if isinstance(data, np.ndarray)
        else data
    )

    # Plot table with matplotlib
    fig, ax = plt.subplots(figsize=figsize)  # set size frame
    ax.axis('off')
    tbl = pd.plotting.table(ax, df, loc='center', cellLoc='center', rowLoc='center')

    # Highlight the cells based on their value with a color map
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)

    return tbl


def format_contingency_table(
    contingency_table: NDArray,
    columns: list[str],
    index: list[str],
    figsize: tuple[int, int] = (12, 3),
    diagonal_color: str = 'darkred',
) -> matplotlib.table.Table:
    contingency_df = pd.DataFrame(
        contingency_table,
        columns=columns,
        index=index,
    )

    # Plot table with matplotlib
    fig, ax = plt.subplots(figsize=figsize)  # set size frame
    ax.axis('off')
    tbl = pd.plotting.table(
        ax, contingency_df, loc='center', cellLoc='center', rowLoc='center'
    )

    # Highlight the cells based on their value with a color map
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)

    # Color grade cell from white to blue based on how large the value is
    for key, cell in tbl.get_celld().items():
        if key[0] == 0 or key[1] == -1:
            # Skip the first row (headers) and first column (index)
            continue

        value = cell.get_text().get_text()

        if value:
            value = float(value)

            # Scale color based on the max value
            color = plt.cm.Blues(value / contingency_df.values.max())
            cell.set_facecolor(color)

            if value > contingency_df.values.max() / 2:
                # If cell value is large, and the cell is really blue, then make the text white
                cell.get_text().set_color('white')

    for i in range(len(contingency_df)):
        # Make the diagonal cells bold and red
        cell = tbl[(i + 1, i)]
        cell.set_text_props(weight='bold', color=diagonal_color)

    return tbl
