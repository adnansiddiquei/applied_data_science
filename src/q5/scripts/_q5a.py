from multiprocessing import Pool
from typing import Literal
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from src.utils import (
    format_axes,
    save_fig,
    compute_num_cores_to_utilise,
    format_contingency_table,
    save_dict_to_json,
)


def _optimal_n_init_kmeans_parallel(X, n_init, n_clusters=2, iters=5):
    return np.array(
        [
            KMeans(n_clusters=n_clusters, n_init=n_init, random_state=j * n_init)
            .fit(X)
            .inertia_
            for j in range(iters)
        ]
    )


def optimal_n_init_kmeans(X, n_init_range, n_clusters=2, iters=3):
    X = StandardScaler().fit_transform(X)

    with Pool(compute_num_cores_to_utilise()) as pool:
        inertia_scores = np.array(
            pool.starmap(
                _optimal_n_init_kmeans_parallel,
                [(X, n_init, n_clusters, iters) for n_init in n_init_range],
            )
        )

    return pd.DataFrame(
        {
            'means': inertia_scores.mean(axis=1),
            'stds': inertia_scores.std(axis=1),
        },
        index=n_init_range,
    )


def _optimal_n_init_gmm_parallel(X, n_init, n_components=2, iters=5):
    return np.array(
        [
            GaussianMixture(
                n_components=n_components, n_init=n_init, random_state=j * n_init
            )
            .fit(X)
            .score(X)
            for j in range(iters)
        ]
    )


def optimal_n_init_gmm(X, n_init_range, n_components=2, iters=5):
    X = StandardScaler().fit_transform(X)

    with Pool(compute_num_cores_to_utilise()) as pool:
        log_likelihoods = np.array(
            pool.starmap(
                _optimal_n_init_gmm_parallel,
                [(X, n_init, n_components, iters) for n_init in n_init_range],
            )
        )

    return pd.DataFrame(
        {
            'means': log_likelihoods.mean(axis=1),
            'stds': log_likelihoods.std(axis=1),
        },
        index=n_init_range,
    )


def _compute_silhouette_score_parallel(model, X, n_clusters, n_init, iters=10):
    def get_instantiated_model(model, n_clusters, random_state):
        if model == 'kmeans':
            return KMeans(
                n_clusters=n_clusters, random_state=random_state, n_init=n_init
            )
        elif model == 'gmm':
            return GaussianMixture(
                n_components=n_clusters, random_state=random_state, n_init=n_init
            )
        else:
            raise ValueError('Model must be either KMeans or GaussianMixture')

    scores = [
        silhouette_score(
            X,
            get_instantiated_model(
                model, n_clusters, i * n_clusters * len(model)
            ).fit_predict(X),
        )
        for i in range(iters)
    ]

    return scores


def compute_silhouette_score(
    model: Literal['kmeans', 'gmm'], X, cluster_range, n_init=1, iters=5
):
    X = StandardScaler().fit_transform(X)

    with Pool(compute_num_cores_to_utilise()) as pool:
        scores = np.array(
            pool.starmap(
                _compute_silhouette_score_parallel,
                [(model, X, n_clusters, n_init, iters) for n_clusters in cluster_range],
            )
        )

    return pd.DataFrame(
        {
            'means': scores.mean(axis=1),
            'stds': scores.std(axis=1),
        },
        index=cluster_range,
    )


def predict_and_relabel(X, gmm: GaussianMixture, kmeans: KMeans):
    y_pred_gmm = gmm.fit_predict(X)
    y_pred_kmeans = kmeans.fit_predict(X)

    distances = pairwise_distances(gmm.means_, kmeans.cluster_centers_)
    row_ind, col_ind = linear_sum_assignment(distances)

    y_pred_kmeans = (
        pd.Series(y_pred_kmeans)
        .replace({col_ind[i]: row_ind[i] for i in range(len(row_ind))})
        .values
    )

    return y_pred_gmm, y_pred_kmeans


def q5a():
    # Load the data
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        f'{cwd}/../../q4/outputs/ADS_baselineDataset_preprocessed.csv', index_col=0
    )

    data, classifications = data[data.columns[:-1]], data['type']

    X = data.copy().values
    y = classifications.copy().values

    # Compute the silhouette score for the GMM and KMeans, to estimate optimal number of clusters
    kmeans_silhouette_scores = compute_silhouette_score('kmeans', X, range(2, 10))
    gmm_silhouette_scores = compute_silhouette_score('gmm', X, range(2, 10))

    fig, ax = plt.subplots()

    plt.errorbar(
        kmeans_silhouette_scores.index,
        kmeans_silhouette_scores['means'],
        yerr=kmeans_silhouette_scores['stds'],
        label='KMeans',
        marker='o',
        markersize=5,
        capsize=3,
    )

    plt.errorbar(
        gmm_silhouette_scores.index + 0.1,
        gmm_silhouette_scores['means'],
        yerr=gmm_silhouette_scores['stds'],
        label='GMM',
        color='darkorange',
        alpha=0.7,
        marker='o',
        markersize=5,
        capsize=3,
    )

    plt.ylabel('Silhouette Score')
    plt.xlabel('Number of Clusters')

    format_axes(ax)
    plt.legend()

    save_fig(__file__, 'q5a_silhouette_scores.png')

    # Now we compute optimal n_init to converge on a good number of clusters
    kmeans_n_init = optimal_n_init_kmeans(
        X, [1] + list(range(2, 11, 2)) + list(range(14, 25, 4))
    )
    gmm_n_init = optimal_n_init_gmm(
        X, [1] + list(range(2, 11, 2)) + list(range(14, 25, 4))
    )

    fig, ax = plt.subplots()

    plt.errorbar(
        kmeans_n_init.index,
        kmeans_n_init['means'],
        yerr=kmeans_n_init['stds'],
        marker='o',
        markersize=5,
        capsize=3,
        label='Inertia Score',
    )

    plt.ylabel('Inertia Score (KMeans)')
    ax2 = ax.twinx()

    ax2.errorbar(
        gmm_n_init.index,
        gmm_n_init['means'],
        yerr=gmm_n_init['stds'],
        marker='o',
        markersize=5,
        capsize=3,
        label='Log Likelihood',
        color='darkorange',
        alpha=0.7,
    )

    ax2.set_ylabel('Log Likelihood (GMM)')

    format_axes([ax, ax2], combine_legends=True)

    plt.xlabel(r'Number of Initializations $n\_init$')

    save_fig(__file__, 'q5a_n_init_optimisation.png')

    # Now let's compute the clusterings
    gmm = GaussianMixture(n_components=2, n_init=50, random_state=3438)
    kmeans = KMeans(n_clusters=2, n_init=50, random_state=8343)

    y_pred_gmm, y_pred_kmeans = predict_and_relabel(X, gmm, kmeans)

    cmatrix = pd.DataFrame(confusion_matrix(y_pred_gmm, y_pred_kmeans))
    cmatrix[3] = cmatrix.sum(axis=1)
    cmatrix.loc[3] = cmatrix.sum(axis=0)

    tbl = format_contingency_table(
        cmatrix.values,
        columns=['1', '2', 'Tot. (GMM)'],
        index=['1', '2', 'Tot. (KMeans)'],
        figsize=(3, 3),
    )

    tbl.scale(1.0, 1.2)

    tbl[(3, 2)].set_facecolor('white')
    tbl[(3, 2)].set_text_props(color='black')

    save_fig(__file__, 'q5a_contingency_table.png')

    # And save the classifications so they can be used in the next part
    results_dict = {
        'GMM': list(map(int, y_pred_gmm)),
        'KMeans': list(map(int, y_pred_kmeans)),
    }

    save_dict_to_json(results_dict, __file__, 'q5a_classifications.json')
