from .q1utils import kmeans_on_dataset_a
from src.utils import format_axes, load_dataset, save_fig
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


def q1e():
    # Perform the k-means clustering to get the clusters from each model: kmeans_1_all, kmeans_2_all
    _contingency_table, tbl, kmeans_1_all, kmeans_2_all = kmeans_on_dataset_a(
        n_clusters=3, random_state=3438
    )

    # Load the dataset
    data = load_dataset(
        'A_NoiseAdded.csv',
        drop_columns=['Unnamed: 0', 'classification'],
        standardise=True,
    )

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(data)

    # Compute the scores on each observation
    z = pd.DataFrame(pca.transform(data))

    # Add the classifications to the scores DataFrame
    z['classification_1'] = kmeans_1_all
    z['classification_2'] = kmeans_2_all

    # Plot the PCA, colour coded by the classifications
    color_map = {0: 'blue', 1: 'red', 2: 'green'}
    fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)

    for classification, color in color_map.items():
        subset = z[z['classification_1'] == classification]
        ax[0].scatter(
            subset[0],
            subset[1],
            c=color,
            label=f'Cluster {classification + 1}',
            alpha=0.5,
        )

    for classification, color in color_map.items():
        subset = z[z['classification_2'] == classification]
        ax[1].scatter(
            subset[0],
            subset[1],
            c=color,
            label=f'Cluster {classification + 1}',
            alpha=0.5,
        )

    # Format the plots
    ax[0].legend()
    ax[1].legend()

    format_axes(ax[0])
    format_axes(ax[1])

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')

    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC2')

    ax[0].set_title(r'$kmeans\_1$')
    ax[1].set_title(r'$kmeans\_2$')

    save_fig(__file__, 'q1e.png')
