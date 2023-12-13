from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from src.utils import load_dataset, format_contingency_table
import matplotlib.pyplot as plt
import os

import warnings

warnings.filterwarnings('ignore')


def q1c():
    data = load_dataset('A_NoiseAdded.csv')
    data = data.drop(['Unnamed: 0'], axis=1)

    # Split the data into two equal-sized samples
    train_set_1, train_set_2 = train_test_split(data, test_size=0.5, random_state=3438)

    # Fit a k-means model on each training set
    kmeans_1 = KMeans(random_state=3438)
    kmeans_2 = KMeans(random_state=3438)

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

    # Create the contingency table
    contingency_table = confusion_matrix(kmeans_1_all.values, kmeans_2_all.values)

    format_contingency_table(
        contingency_table,
        columns=[f'Cluster {i + 1}' for i in range(contingency_table.shape[1])],
        index=[f'Cluster {i + 1}' for i in range(contingency_table.shape[0])],
    )

    cwd = os.path.dirname(os.path.realpath(__file__))

    plt.savefig(
        os.path.join(cwd, '../outputs/q1c-contingency-table.png'),
        bbox_inches='tight',
        dpi=500,
    )
