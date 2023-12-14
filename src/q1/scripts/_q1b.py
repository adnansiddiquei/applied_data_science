from sklearn.decomposition import PCA
from src.utils import load_dataset, format_axes
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colormaps as cmaps
from matplotlib.patches import FancyArrowPatch


def q1b():
    data = load_dataset(
        'A_NoiseAdded.csv',
        drop_columns=['Unnamed: 0', 'classification'],
        standardise=True,
    )

    pca = PCA(n_components=2)
    pca.fit(data)

    # Compute the scores on each observation
    z = pca.transform(data)

    # Plot the scores
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-22, 22)
    plt.plot(z[:, 0], z[:, 1], 'o', ms=4)

    # Plot the loadings
    ax2 = ax.twinx().twiny()
    ax2.set_xlim(-0.3, 0.3)
    ax2.set_ylim(-0.3, 0.3)

    # Transpose makes it easier to plot
    loadings = pca.components_.T

    # Draw arrows from (0,0) to the end of each loading vector

    # Draw arrows for only loadings that contribute >2% to either PC
    pct = 0.02

    # This is the number of loadings that contribute more than 3%
    count = np.sum(
        [loading[0] ** 2 > pct or loading[1] ** 2 > pct for loading in loadings]
    )

    # These are colours for each of them
    colours = (elem for elem in [cmaps['hsv'](x) for x in np.linspace(0, 1, count + 1)])

    for i, loading in enumerate(loadings):
        if loading[0] ** 2 > pct or loading[1] ** 2 > pct:
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
        elif i < 20:
            if i + 1 in [5, 18, 19, 20, 14, 11, 13]:
                arrow = FancyArrowPatch(
                    (0, 0),
                    (loading[0], loading[1]),
                    color='k',
                    linestyle=(0, (1, 3)),
                    linewidth=1.3,
                )
                ax2.add_patch(arrow)
            else:
                arrow = FancyArrowPatch(
                    (0, 0),
                    (loading[0], loading[1]),
                    color='k',
                    linewidth=1,
                )
                ax2.add_patch(arrow)
        else:
            plt.arrow(
                0,
                0,
                loading[0],
                loading[1],
                color='gray',
                alpha=0.06,
                head_width=0.006,
            )

    # Format the axes
    ax2.legend()
    format_axes(ax, ticks_right=False, ticks_top=False)
    format_axes(ax2, ticks_bottom=False, ticks_left=False, legend_loc='upper left')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # Split the legends into two parts
    # split = 20
    # handles, labels = plt.gca().get_legend_handles_labels()
    # first_legend = plt.legend(
    #     handles=handles[:split], labels=labels[:split], loc='best', facecolor='white'
    # )
    # plt.gca().add_artist(first_legend)
    # plt.legend(
    #     handles=handles[split:], labels=labels[split:], loc='lower left', facecolor='white'
    # )

    cwd = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(cwd, '../outputs/q1b.png'), bbox_inches='tight')
