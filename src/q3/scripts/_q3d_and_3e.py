import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from src.utils import format_axes, save_fig
from sklearn.impute import KNNImputer
from numpy.typing import NDArray


def compute_outliers(data: np.ndarray, z_threshold: float = 3.0):
    data = data.copy()

    z_scores = StandardScaler().fit_transform(data)

    # A 2d numpy array of booleans indicating whether a value is an outlier
    outliers: NDArray[bool] = (np.abs(z_scores) > z_threshold) & (
        np.broadcast_to(data.var(axis=0), z_scores.shape) > 1e-8
    )

    # The percentage of outliers in the dataset
    pct_outliers: float = np.sum(outliers) / np.prod(outliers.shape)

    return outliers, z_scores, pct_outliers


def knn_impute_nans(
    data: pd.DataFrame, classifications: pd.Series, n_neighbors: int = 15
):
    """Impute NaNs in a DataFrame using KNNImputer on each classification separately."""
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Impute the missing values using KNNImputer, with optimal value computed in q3c_optimise_knn_imputer.py
    knn_imputer = KNNImputer(n_neighbors=15)

    # This will store "data" with the missing values imputed
    knn_imputed_data = pd.DataFrame(columns=data.columns)

    # Loop through each classification and impute the missing values, using KNN
    for c in classifications.unique():
        # Get the data that belongs to classification c
        data_in_classification = data[classifications == c]

        # Impute the missing values with KNN
        imputed_data_in_classification = pd.DataFrame(
            knn_imputer.fit_transform(data_in_classification),
            index=data_in_classification.index,
            columns=data_in_classification.columns,
        )

        knn_imputed_data = pd.concat(
            [knn_imputed_data, imputed_data_in_classification], axis=0
        )

    knn_imputed_data = knn_imputed_data.sort_index()

    # Invert the standardisation
    knn_imputed_data = pd.DataFrame(
        scaler.inverse_transform(knn_imputed_data), columns=data.columns
    )

    return knn_imputed_data


def knn_impute_outliers(data: pd.DataFrame, classifications: pd.Series):
    """Impute outliers in a DataFrame using KNNImputer on each classification separately."""
    data = data.copy()
    outliers, z_scores, pct_outliers = compute_outliers(data.values)

    data[outliers] = np.nan

    outliers_imputed = knn_impute_nans(data, classifications)

    new_outliers, new_z_scores, new_pct_outliers = compute_outliers(
        outliers_imputed.values
    )

    return outliers_imputed, new_outliers, new_z_scores, new_pct_outliers


def iteratively_knn_impute_outliers(
    data: pd.DataFrame, classifications: pd.Series, n_iterations: int = 5
):
    """Keep imputing outliers for a given number of iterations"""
    outliers, z_scores, pct_outliers = compute_outliers(data.values)

    results = pd.DataFrame(
        [[np.sum(outliers), pct_outliers, np.var(z_scores)]],
        columns=['n_outliers', 'pct_outliers', 'variance'],
    )

    for i in range(n_iterations):
        (
            outliers_imputed,
            new_outliers,
            new_z_scores,
            new_pct_outliers,
        ) = knn_impute_outliers(outliers_imputed if i > 0 else data, classifications)

        results.loc[i + 1] = [
            np.sum(new_outliers),
            new_pct_outliers,
            np.var(new_z_scores),
        ]

    return outliers_imputed.round(9), results


def q3d_and_q3e():
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        os.path.join(cwd, '../outputs/knn_imputed_data.csv'), index_col=0
    )
    data, classifications = data[data.columns[:-1]], data['classification']

    # There are a total of 2904 outliers
    outliers, z_scores, pct_outliers = compute_outliers(data.values)

    # Create the heatmap using seaborn
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(outliers, cmap='viridis', cbar=False)
    plt.xlabel('Features')
    plt.ylabel('Samples')
    format_axes(ax)

    save_fig(__file__, 'q3d_heatmap.png')

    # Now impute the outliers using KNNImputer
    new_data, imputation_data = iteratively_knn_impute_outliers(
        data, classifications, n_iterations=44
    )

    # Plot the results
    # Plot the results
    fig, ax = plt.subplots()
    plt.xlabel('Iteration')

    plt.errorbar(
        x=imputation_data.index,
        y=imputation_data['variance'],
        marker='o',
        markersize=4,
        label='Variance of Dataset',
    )

    ax2 = ax.twinx()

    plt.errorbar(
        x=imputation_data.index,
        y=imputation_data['pct_outliers'],
        marker='o',
        c='orange',
        markersize=4,
        label='% of Dataset that are Outliers',
        alpha=0.7,
    )

    format_axes(ax, ticks_right=False)
    format_axes(ax2, ticks_left=False)

    # Collect the legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    handles.extend(handles2)
    labels.extend(labels2)

    # into  a single legend
    ax.legend(handles, labels)

    ax.set_ylabel('Variance')
    ax2.set_ylabel('% Outliers')

    save_fig(__file__, 'q3e_imputation_data.png')
