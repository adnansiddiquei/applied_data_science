import matplotlib.pyplot as plt
import numpy as np
from src.utils import format_axes, load_dataset, save_fig
from sklearn.cluster import KMeans


def q1a():
    """Q1a

    Generate density plots for the first 20 features. Include the figure in your report and state what you observe.
    """
    # Load the dataset
    data = load_dataset(
        'A_NoiseAdded.csv', drop_columns=['Unnamed: 0', 'classification']
    )

    # Split off the first 20 features, then we will group them by their variance to plot them separately
    feats20 = data[data.columns[0:20]]
    var_feats20 = np.array(feats20.var())

    n_clusters = 2
    groups = KMeans(n_clusters=n_clusters, random_state=3438).fit_predict(
        var_feats20.reshape(-1, 1)
    )

    # Now create the plot, we will do 2 plots, as we have split the data into 2 groups
    fig, ax = plt.subplots(figsize=(10, 6 * n_clusters), nrows=n_clusters, ncols=1)

    for c, group in zip(feats20.columns, groups):
        feats20[c].plot(kind='density', ax=ax[group], label=c)

    [ax[i].legend() for i in range(n_clusters)]
    [format_axes(ax[i]) for i in range(n_clusters)]
    [ax[i].set_ylim(-0.1, 2.8) for i in range(n_clusters)]
    [ax[i].set_xlim(-2, 8) for i in range(n_clusters)]

    ax[0].title.set_text(
        f'Lower Variance Features ({var_feats20[np.where(groups == 0)].min().round(2)} < Var < {var_feats20[np.where(groups == 0)].max().round(2)})'
    )
    ax[1].title.set_text(
        f'Higher Variance Features ({var_feats20[np.where(groups == 1)].min().round(2)} < Var < {var_feats20[np.where(groups == 1)].max().round(2)})'
    )

    plt.autoscale(enable=True, axis='x', tight=True)

    save_fig(__file__, 'q1a.png')
