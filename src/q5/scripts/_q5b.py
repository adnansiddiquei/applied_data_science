import os
import pandas as pd
from src.utils import (
    compute_most_important_features_logit,
    load_dict_from_json,
    compute_rolling_intersection_pct,
    plot_feature_importance,
    format_axes,
    save_fig,
    format_contingency_table,
    save_dict_to_json,
    save_dataframe_to_csv,
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def q5b():
    # Load the data
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        f'{cwd}/../../q4/outputs/ADS_baselineDataset_preprocessed.csv', index_col=0
    )

    data, classifications = data[data.columns[:-1]], data['type']

    X = data.copy().values
    y = classifications.copy().values

    X_scaled = StandardScaler().fit_transform(X)

    # Load the classifications predicted from q5a
    predicted_classifications = load_dict_from_json(
        __file__, 'q5a_classifications.json'
    )
    y_pred_gmm_q5a = predicted_classifications['GMM']
    y_pred_kmeans_q5a = predicted_classifications['KMeans']

    # Compute the most important features for the GMM and KMeans predicted classifications, and plot it
    (
        feature_importance_gmm,
        most_importance_features_gmm,
    ) = compute_most_important_features_logit(X_scaled, y_pred_gmm_q5a)

    (
        feature_importance_kmeans,
        most_importance_features_kmeans,
    ) = compute_most_important_features_logit(X_scaled, y_pred_kmeans_q5a)

    rolling_overlap = (
        compute_rolling_intersection_pct(
            feature_importance_gmm['feature'].values,
            feature_importance_kmeans['feature'].values,
        )
        * 100
    )

    fig, ax = plot_feature_importance(
        feature_importance_gmm,
        most_importance_features_gmm,
        label='NMSLI GMM',
        draw_intersection=False,
    )

    plt.plot(
        feature_importance_kmeans['cumulative_importance'],
        label='NMSLI KMeans',
        color='forestgreen',
        linestyle=':',
    )

    plt.ylabel('Cum. Norm. Mean Squared Logit Importance (NMSLI)')
    plt.axhline(y=0.95, color='grey', linestyle='--')

    # Add the vertical lines and text for the number of features selected
    plt.axvline(
        x=len(most_importance_features_gmm), color='#1f77b4', linestyle='--', alpha=0.5
    )
    plt.axvline(
        x=len(most_importance_features_kmeans),
        color='forestgreen',
        linestyle='--',
        alpha=0.5,
    )

    plt.text(
        len(most_importance_features_gmm) + 12,
        0.02,
        f'{len(most_importance_features_gmm)}',
        fontsize=12,
        color='#1f77b4',
        ha='left',
        va='bottom',
    )

    plt.text(
        len(most_importance_features_kmeans) - 12,
        0.02,
        f'{len(most_importance_features_kmeans)}',
        fontsize=12,
        color='forestgreen',
        ha='right',
        va='bottom',
    )

    plt.text(
        980,
        0.93,
        '95% cum. NMSLI',
        ha='right',
        va='top',
        fontsize=12,
        color='grey',
    )

    ax2 = ax.twinx()
    ax2.plot(rolling_overlap, label='Rolling Feature Overlap %', color='darkorange')
    ax2.set_ylabel('Rolling Feature Overlap %')
    ax2.set_ylim(-5, 105)

    format_axes([ax, ax2], combine_legends=True)

    crossover_gmm = rolling_overlap[len(most_importance_features_gmm)]
    crossover_kmeans = rolling_overlap[len(most_importance_features_kmeans)]
    avg_crossover = np.mean([crossover_gmm, crossover_kmeans])

    plt.axhline(avg_crossover, color='purple', linestyle='--', alpha=0.5)

    plt.text(
        980,
        avg_crossover - 2,
        rf'$\approx${
        (avg_crossover / 100) * np.mean([len(most_importance_features_gmm), len(most_importance_features_kmeans)]):.0f} ({avg_crossover:.0f}%) overlapping'
        + '\nfeatures at 95% cum. NMSLI',
        ha='right',
        va='top',
        fontsize=10,
        color='purple',
    )

    ax.autoscale(enable=True, tight=True, axis='x')

    save_fig(__file__, 'q5b_feature_importance.png')

    # Now we compute the clusterings with only the most important features
    X_subset_gmm = data[most_importance_features_gmm['feature']].copy().values
    X_subset_kmeans = data[most_importance_features_kmeans['feature']].copy().values

    X_subset_gmm_scaled = StandardScaler().fit_transform(X_subset_gmm)
    X_subset_kmeans_scaled = StandardScaler().fit_transform(X_subset_kmeans)

    # Now let's compute the clusterings
    gmm = GaussianMixture(n_components=2, n_init=50, random_state=3438)
    kmeans = KMeans(n_clusters=2, n_init=50, random_state=8343)

    y_pred_gmm = gmm.fit_predict(X_subset_gmm_scaled)
    y_pred_kmeans = kmeans.fit_predict(X_subset_kmeans_scaled)

    # We have to re-label, in the case that the labels have been flipped from q5a
    if np.sum(y_pred_gmm == y_pred_gmm_q5a) / len(y_pred_gmm) < 0.5:
        y_pred_gmm = pd.Series(y_pred_gmm).replace({0: 1, 1: 0}).values

    if np.sum(y_pred_kmeans == y_pred_kmeans_q5a) / len(y_pred_kmeans) < 0.5:
        y_pred_kmeans = pd.Series(y_pred_kmeans).replace({0: 1, 1: 0}).values

    cmatrix = pd.DataFrame(confusion_matrix(y_pred_gmm, y_pred_kmeans))
    cmatrix[3] = cmatrix.sum(axis=1)
    cmatrix.loc[3] = cmatrix.sum(axis=0)

    tbl = format_contingency_table(
        cmatrix.values,
        columns=['1', '2', 'Tot. (GMM)'],
        index=['1', '2', 'Tot. (KMeans)'],
        figsize=(3, 3),
    )

    tbl.scale(1.2, 1.2)

    tbl[(3, 2)].set_facecolor('white')
    tbl[(3, 2)].set_text_props(color='black')

    save_fig(__file__, 'q5b_contingency_table.png')

    # And save the classifications so they can be used in the next part
    results_dict = {
        'GMM': list(map(int, y_pred_gmm)),
        'KMeans': list(map(int, y_pred_kmeans)),
    }

    save_dict_to_json(results_dict, __file__, 'q5b_classifications.json')

    # And save the most important features so the next part can access it
    save_dataframe_to_csv(
        most_importance_features_gmm,
        __file__,
        'q5b_most_importance_features_gmm.csv',
        index=False,
    )
    save_dataframe_to_csv(
        most_importance_features_kmeans,
        __file__,
        'q5b_most_importance_features_kmeans.csv',
        index=False,
    )
