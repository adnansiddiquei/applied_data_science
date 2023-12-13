from sklearn.decomposition import PCA
from src.utils import load_dataset, format_axes
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def q1b():
    data = load_dataset('A_NoiseAdded.csv')
    data = data.drop(['Unnamed: 0'], axis=1)

    features = data.columns[0:20]
    X = data[features]

    pca = PCA(n_components=2)
    pca.fit(X)

    # Compute the scores on each observation
    z = pca.transform(X)

    # Plot the scores
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(z[:, 0], z[:, 1], 'o', ms=4)

    # Plot the loadings
    ax2 = ax.twinx().twiny()
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-0.8, 0.8)

    # Transpose makes it easier to plot
    loadings = pca.components_.T

    # Draw arrows from (0,0) to the end of each loading vector, random seed is for colour
    np.random.seed(15)
    for i, loading in enumerate(loadings):
        # Only plot the loading if it contributes more than 1%
        if abs(loading[0] ** 2) > 0.01 or abs(loading[1] ** 2) > 0.01:
            # The following loadings contribute more than 1%
            c = {
                5: 'b',
                11: 'y',
                13: 'gray',
                14: 'm',
                18: 'g',
                19: 'r',
                20: 'c',
                15: 'orange',
                17: 'purple',
            }

            plt.arrow(
                0,
                0,
                loading[0],
                loading[1],
                color=c[i + 1],
                head_width=0.02,
                label=f'Fea{i + 1}',
            )

    # Format the axes
    ax2.legend()
    format_axes(ax, ticks_right=False, ticks_top=False)
    format_axes(ax2, ticks_bottom=False, ticks_left=False)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    cwd = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(cwd, '../outputs/q1b-1.png'), bbox_inches='tight')

    # Now we plot the explained variance
    pca = PCA(n_components=12)
    pca.fit(X)

    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 2))
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    sns.lineplot(x=range(1, 13), y=explained_variance, ax=ax, marker='o', ms=5)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim(0, 1)
    plt.axvline(2, color='k', linestyle='--', alpha=0.5)

    plt.axhline(explained_variance[1], color='k', linestyle='--', alpha=0.5)
    format_axes(ax)

    plt.savefig(os.path.join(cwd, '../outputs/q1b-2.png'), bbox_inches='tight')
