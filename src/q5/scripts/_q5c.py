import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import load_dict_from_json, load_dataframe_from_csv, save_fig
import numpy as np
from sklearn.decomposition import PCA


def q5c():
    # Load the data
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        f'{cwd}/../../q4/outputs/ADS_baselineDataset_preprocessed.csv', index_col=0
    )

    data, classifications = data[data.columns[:-1]], data['type']

    X = data.copy().values
    y = classifications.copy().values

    X_scaled = StandardScaler().fit_transform(X)

    predicted_classifications = load_dict_from_json(
        __file__, 'q5b_classifications.json'
    )

    y_pred_gmm_q5b = predicted_classifications['GMM']
    y_pred_kmeans_q5b = predicted_classifications['KMeans']

    gmm_important_features = load_dataframe_from_csv(
        __file__, 'q5b_most_importance_features_gmm.csv'
    )

    kmeans_important_features = load_dataframe_from_csv(
        __file__, 'q5b_most_importance_features_kmeans.csv'
    )

    X_subset_gmm = data[gmm_important_features['feature']].copy().values
    X_subset_kmeans = data[kmeans_important_features['feature']].copy().values

    X_subset_scaled_gmm = StandardScaler().fit_transform(X_subset_gmm)
    X_subset_scaled_kmeans = StandardScaler().fit_transform(X_subset_kmeans)

    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    z = pca.transform(X_scaled)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    colorbar_min = np.concatenate(
        [X_subset_scaled_gmm[:, 2], X_subset_scaled_kmeans[:, 2]]
    ).min()

    colorbar_max = np.concatenate(
        [X_subset_scaled_gmm[:, 2], X_subset_scaled_kmeans[:, 2]]
    ).max()

    color_map = {0: 'darkorange', 1: 'black'}

    s1 = ax[0, 0].scatter(
        z[:, 0], z[:, 1], c=[color_map[label] for label in y_pred_gmm_q5b], s=10
    )

    s2 = ax[0, 1].scatter(
        z[:, 0],
        z[:, 1],
        c=X_subset_scaled_gmm[:, 0],
        s=10,
        cmap='seismic',
        vmin=colorbar_min,
        vmax=colorbar_max,
    )

    s3 = ax[0, 2].scatter(
        z[:, 0],
        z[:, 1],
        c=X_subset_scaled_gmm[:, 1],
        s=10,
        cmap='seismic',
        vmin=colorbar_min,
        vmax=colorbar_max,
    )

    s4 = ax[1, 0].scatter(
        z[:, 0],
        z[:, 1],
        c=[color_map[label] for label in y_pred_kmeans_q5b],
        s=10,
        cmap='flag',
    )

    s5 = ax[1, 1].scatter(
        z[:, 0],
        z[:, 1],
        c=X_subset_scaled_kmeans[:, 0],
        s=10,
        cmap='seismic',
        vmin=colorbar_min,
        vmax=colorbar_max,
    )
    s6 = ax[1, 2].scatter(
        z[:, 0],
        z[:, 1],
        c=X_subset_scaled_kmeans[:, 1],
        s=10,
        cmap='seismic',
        vmin=colorbar_min,
        vmax=colorbar_max,
    )

    # Create an axis for the colorbar on the right side of the figure
    fig.subplots_adjust(right=0.85)  # Adjust the subplot to make room for the colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Position for the colorbar

    # Add the colorbar
    cbar = fig.colorbar(s2, cax=cbar_ax)

    ax[0, 0].text(10, 10, 'GMM', ha='right', va='top', fontsize=12, color='grey')
    ax[1, 0].text(10, 10, 'KMeans', ha='right', va='top', fontsize=12, color='grey')

    ax[0, 0].text(-9, 8.5, 'Class 1', ha='left', va='top', fontsize=10, color='grey')
    ax[0, 0].text(10, -9, 'Class 2', ha='right', va='bottom', fontsize=10, color='grey')

    ax[0, 1].text(
        10,
        10,
        f'{gmm_important_features["feature"].iloc[0]}',
        ha='right',
        va='top',
        fontsize=12,
        color='grey',
    )

    ax[0, 2].text(
        10,
        10,
        f'{gmm_important_features["feature"].iloc[1]}',
        ha='right',
        va='top',
        fontsize=12,
        color='grey',
    )

    ax[1, 1].text(
        10,
        10,
        f'{kmeans_important_features["feature"].iloc[0]}',
        ha='right',
        va='top',
        fontsize=12,
        color='grey',
    )

    ax[1, 2].text(
        10,
        10,
        f'{kmeans_important_features["feature"].iloc[1]}',
        ha='right',
        va='top',
        fontsize=12,
        color='grey',
    )

    save_fig(__file__, 'q5c.png')
