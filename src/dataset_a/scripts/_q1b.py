from sklearn.decomposition import PCA
from src.utils import load_dataset, format_axes
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colormaps as cmaps


def q1b():
    data = load_dataset('A_NoiseAdded.csv')
    data = data.drop(['Unnamed: 0'], axis=1)

    pca = PCA(n_components=2)
    pca.fit(data)

    # Compute the scores on each observation
    z = pca.transform(data)

    # Plot the scores
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(z[:, 0], z[:, 1], 'o', ms=4)

    # Plot the loadings
    ax2 = ax.twinx().twiny()
    ax2.set_xlim(-0.14, 0.14)
    ax2.set_ylim(-0.5, 0.5)

    # Transpose makes it easier to plot
    loadings = pca.components_.T

    # Draw arrows from (0,0) to the end of each loading vector

    # Draw arrows for only loadings the contribute >3% to the PC
    pct = 0.025

    # This is the number of loadings that contribute more than 3%
    count = np.sum(
        [
            abs(loading[0] ** 2) > pct or abs(loading[1] ** 2) > pct
            for loading in loadings
        ]
    )

    # These are colours for each of them
    colours = (elem for elem in [cmaps['hsv'](x) for x in np.linspace(0, 1, count)])

    for i, loading in enumerate(loadings):
        if abs(loading[0] ** 2) > pct or abs(loading[1] ** 2) > pct:
            # The following loadings contribute more than 3%
            plt.arrow(
                0,
                0,
                loading[0],
                loading[1],
                color=next(colours),
                head_width=0.006,
                label=f'Fea{i + 1}',
            )

        # These are the first 20 loadings
        if i < 20:
            if i + 1 in [5, 18, 19, 20, 14, 11, 13]:
                plt.arrow(
                    0,
                    0,
                    loading[0],
                    loading[1],
                    color='k',
                    head_width=0.003,
                    linewidth=0.2,
                    label=f'Fea{i + 1}',
                )
            else:
                plt.arrow(
                    0,
                    0,
                    loading[0],
                    loading[1],
                    color='gray',
                    head_width=0.003,
                    linewidth=0.2,
                    label=f'Fea{i + 1}',
                )

    # Format the axes
    # ax2.legend()
    format_axes(ax, ticks_right=False, ticks_top=False)
    format_axes(ax2, ticks_bottom=False, ticks_left=False)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # Split the legends into two parts
    handles, labels = plt.gca().get_legend_handles_labels()
    first_legend = plt.legend(
        handles=handles[:20], labels=labels[:20], loc='upper right', facecolor='white'
    )
    plt.gca().add_artist(first_legend)
    plt.legend(
        handles=handles[20:], labels=labels[20:], loc='lower left', facecolor='white'
    )

    cwd = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(cwd, '../outputs/q1b.png'), bbox_inches='tight')
