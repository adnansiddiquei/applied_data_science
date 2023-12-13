import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from ...utils import format_axes


def q1a():
    # Get the current working directory of this file
    cwd = os.path.dirname(os.path.realpath(__file__))

    # Read the dataset
    data = pd.read_csv(os.path.join(cwd, '../../datasets/A_NoiseAdded.csv'))
    data = data.drop(['Unnamed: 0'], axis=1)

    # Plot the KDE of the first 20 Features
    features = data.columns[0:20]

    # Create a colourmap for the features, for the line plots
    colormap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=len(features) - 1)
    colors = [colormap(norm(value)) for value in range(len(features))]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax2 = ax.twinx()

    # Plot the total KDE of all 20 features combined
    combined_data = pd.Series(
        np.concatenate([data[feature].values for feature in features])
    )
    combined_data.plot(
        kind='density', linestyle='--', color='black', ax=ax, label='Combined KDE'
    )

    # Plot the KDE of the first 20 features
    for i, feature in enumerate(features):
        if i + 1 in [5, 18, 19, 20, 14, 11]:
            c = {
                5: 'red',
                18: 'blue',
                19: 'orange',
                20: 'purple',
                14: 'aquamarine',
                11: 'slategray',
            }
            data[feature].plot(
                kind='density', label=f'* {feature}', color=c[i + 1], ax=ax2
            )
        else:
            data[feature].plot(kind='density', label=feature, color=colors[i], ax=ax2)

    plt.xlabel('Value')

    ax.set_ylabel('Kernel Density Estimate')
    ax2.set_ylabel('')

    ax.legend()
    ax2.legend()

    # Format the axes to same the style used throughout the report
    format_axes(ax, ticks_right=False, legend_loc='upper left')
    format_axes(ax2, ticks_left=False, legend_loc='upper right')

    # Scale the x-axis
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlim(-2, 8)

    plt.savefig(os.path.join(cwd, '../outputs/q1a.png'), bbox_inches='tight')
