import matplotlib.table
import pandas as pd
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from src.utils import load_dataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def kmeans_on_dataset_a(n_clusters, random_state):
    data = load_dataset(
        'A_NoiseAdded.csv',
        drop_columns=['Unnamed: 0', 'classification'],
        standardise=True,
    )

    # Split the data into two equal-sized samples
    train_set_1, train_set_2 = train_test_split(
        data, test_size=0.5, random_state=random_state
    )

    # Fit a k-means model on each training set
    kmeans_1 = KMeans(random_state=random_state, n_clusters=n_clusters)
    kmeans_2 = KMeans(random_state=random_state, n_clusters=n_clusters)

    kmeans_1.fit(train_set_1)
    kmeans_2.fit(train_set_2)

    # Predict for the opposite training set
    kmeans_1_preds = kmeans_1.predict(train_set_2)
    kmeans_2_preds = kmeans_2.predict(train_set_1)

    # Combine the training and predictions into a single DataFrame
    kmeans_1_all = (
        pd.DataFrame(
            np.row_stack(
                [
                    np.column_stack([train_set_1.index, kmeans_1.labels_]),
                    np.column_stack([train_set_2.index, kmeans_1_preds]),
                ]
            )
        )
        .set_index(0)
        .sort_index()
    )

    kmeans_2_all = (
        pd.DataFrame(
            np.row_stack(
                [
                    np.column_stack([train_set_2.index, kmeans_2.labels_]),
                    np.column_stack([train_set_1.index, kmeans_2_preds]),
                ]
            )
        )
        .set_index(0)
        .sort_index()
    )

    # Now remap the labels in kmeans_2_all to match those in kmeans_1_all, see docstring of
    # compute_optimal_label_remaps_kmeans function for more info on why we do this
    label_remapping = compute_optimal_label_remaps_kmeans(kmeans_1, kmeans_2)

    kmeans_2_all[1] = kmeans_2_all[1].replace(
        {
            label_remapping[1][i]: label_remapping[0][i]
            for i in range(len(label_remapping[0]))
        }
    )

    # Create the contingency table
    _contingency_table = confusion_matrix(kmeans_1_all.values, kmeans_2_all.values)

    contingency_table = pd.DataFrame(_contingency_table)
    contingency_table.loc[n_clusters] = np.sum(_contingency_table, axis=0)
    contingency_table[n_clusters] = np.append(np.sum(_contingency_table, axis=1), [0])

    tbl = format_contingency_table(
        contingency_table.values,
        columns=[
            f'Cluster {i + 1}' for i in range(contingency_table.values.shape[1] - 1)
        ]
        + ['Total'],
        index=[f'Cluster {i + 1}' for i in range(contingency_table.values.shape[0] - 1)]
        + ['Total'],
    )

    # Make the bottom right cell disappear
    tbl.get_celld()[(n_clusters + 1, n_clusters)].set_text_props(color='white')

    # Make the last row and column bold
    for i in range(len(contingency_table)):
        tbl[(i + 1, n_clusters)].set_text_props(weight='bold')
        tbl[(n_clusters + 1, i)].set_text_props(weight='bold')

    return _contingency_table, tbl, kmeans_1_all, kmeans_2_all


def format_contingency_table(
    contingency_table: NDArray,
    columns: list[str],
    index: list[str],
    figsize: tuple[int, int] = (12, 3),
) -> matplotlib.table.Table:
    contingency_df = pd.DataFrame(
        contingency_table,
        columns=columns,
        index=index,
    )

    # Plot table with matplotlib
    fig, ax = plt.subplots(figsize=(12, 3))  # set size frame
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
        cell.set_text_props(weight='bold', color='darkred')

    return tbl


def compute_optimal_label_remaps_kmeans(
    kmeans_1: KMeans, kmeans_2: KMeans
) -> tuple[NDArray, NDArray]:
    """Given two KMeans models on the same dataset, compute the optimal label remapping algorithm.

    When you fit a KMeans model, the labels are arbitrary. When you want to compare two KMeans models (for example, with
    a contingency table), you need to know which label in kmeans_1 corresponds to which label in kmeans_2. This
    function will compute which labels in kmeans_2 correspond to which labels in kmeans_1 by finding which
    centroids in kmeans_1 map best to which centroids in kmeans_2 (by mapping the closest centroids to each other)

    Parameters
    ----------
    kmeans_1
        A trained KMeans model.
    kmeans_2
        Another trained KMeans model.

    Returns
    -------
    tuple[NDArray, NDArray]
        A tuple of two arrays looking something like ([0, 1, 2], [2, 0, 1]) where the first array lists the labels in
        kmeans_1 and the second array lists the labels in kmeans_2 which map correspond to the same labels in kmeans_1.
    """
    centroids_1 = kmeans_1.cluster_centers_
    centroids_2 = kmeans_2.cluster_centers_

    # compute the pair-wise distances matrix between centroids
    distance_matrix = pairwise_distances(centroids_1, centroids_2)

    # compute the optimal label remapping algorithm, i.e., see which centroid in kmeans_1 maps best to which centroid in
    # kmeans_2
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    return row_ind, col_ind
