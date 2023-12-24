from sklearn.decomposition import PCA
from src.utils import load_dataset, format_axes, save_fig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cmaps
from matplotlib.patches import FancyArrowPatch


def q1b():
    """Q1b

    Apply PCA to visualise the features in 2D.
    """
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

    # Axes labels
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Transpose makes it easier to plot
    loadings = pca.components_.T

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
            if i + 1 in [11, 13, 14, 18, 19, 20]:
                arrow = FancyArrowPatch(
                    (0, 0),
                    (loading[0], loading[1]),
                    color='red',
                    linewidth=1,
                )
                ax2.add_patch(arrow)
            else:
                arrow = FancyArrowPatch(
                    (0, 0),
                    (loading[0], loading[1]),
                    color='forestgreen',
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

    save_fig(__file__, 'q1b.png')
