import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer


def identify_most_discriminative_features(
    X: pd.DataFrame, loading_pct_threshold=0.01
) -> pd.DataFrame:
    """Compute the most discriminative features using PCA.

    Parameters
    ----------
    X
        2D DataFrame of data
    loading_pct_threshold
        The threshold for the percentage of variance explained by a loading for it to be considered a discriminative
        feature

    Returns
    -------
    DataFrame
        A DataFrame of the most discriminative features, with the loadings of each feature on the first two principal
        components, and the index being the feature names.
    """
    X = X.copy()
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    pca = PCA(n_components=2)
    pca.fit(X)
    z = pca.transform(X)

    loadings = pca.components_.T
    loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=X.columns)

    highest_loadings = loadings[
        (loadings['PC1'] ** 2 > loading_pct_threshold)
        | (loadings['PC2'] ** 2 > loading_pct_threshold)
    ]

    return highest_loadings


def knn_impute_nans(X: pd.DataFrame, discriminative_features, n_neighbors=15):
    """
    Impute nan values in a DataFrame using KNNImputer and using only the features in discriminative_features as the
    features to impute from. I.e., when imputing, this function will only use the features in discriminative_features
    and the column that is being imputed to identify the nearest neighbors.

    Parameters
    ----------
    X
        2D DataFrame of data
    discriminative_features
        List of features to use when imputing
    n_neighbors
        Number of nearest neighbors to use when imputing

    Returns
    -------
    DataFrame
        A DataFrame of the imputed data, with the same index and columns as X.
    """
    X = X.copy()
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    imputer = KNNImputer(n_neighbors=n_neighbors)

    imputed_data = pd.DataFrame()

    for column in X.columns:
        feats = list(set([column] + discriminative_features))

        imputed_data[column] = pd.DataFrame(
            imputer.fit_transform(X[feats]), columns=feats
        )[column]

    imputed_data = pd.DataFrame(
        scaler.inverse_transform(imputed_data), columns=X.columns
    ).round(9)

    return imputed_data


def identify_outliers(X: pd.DataFrame, z_threshold=3.0):
    """
    Identify outliers in a DataFrame by computing the z-scores of each value in the DataFrame, and then identifying
    values that are more than z_threshold standard deviations away from the mean.

    Parameters
    ----------
    X
        2D DataFrame of data
    z_threshold
        The threshold for the z-score of a value to be considered an outlier

    Returns
    -------
    NDArray
        A 2D numpy array of booleans indicating whether a value is an outlier. This array has the same shape as X.
    """
    X = X.copy()
    z_scores = StandardScaler().fit_transform(X)

    # A 2d numpy array of booleans indicating whether a value is an outlier
    outliers = (np.abs(z_scores) > z_threshold) & (
        np.broadcast_to(X.var(axis=0), z_scores.shape) > 1e-8
    )

    return outliers


def knn_impute_outliers(
    X: pd.DataFrame, n_neighbors=15, z_threshold=3.0, loading_pct_threshold=0.01
):
    """
    Impute outliers in a DataFrame by first identifying the outliers, then imputing the outliers using KNNImputer.
    This function will only use the most discriminative features to impute the outliers with, to increase the likelihood
    of finding meaningful neighbors.

    Parameters
    ----------
    X
        2D DataFrame of data
    n_neighbors
        Number of nearest neighbors to use when imputing
    z_threshold
        The threshold for the z-score of a value to be considered an outlier
    loading_pct_threshold
        The threshold for the percentage of variance explained by a loading for it to be considered a discriminative
        feature. This is used to compute the most discriminative features.

    Returns
    -------
    tuple[DataFrame, NDArray]
        A DataFrame of the imputed data, with the same index and columns as X. A 2D numpy array of booleans indicating
        where the new outliers in the new dataset are.
    """
    X = X.copy()

    discriminative_features = list(
        identify_most_discriminative_features(X, loading_pct_threshold).index
    )

    outliers = identify_outliers(X, z_threshold=z_threshold)
    X[outliers] = np.nan

    X_with_outliers_imputed = knn_impute_nans(X, discriminative_features, n_neighbors)
    new_outliers = identify_outliers(X_with_outliers_imputed)

    return X_with_outliers_imputed, new_outliers
