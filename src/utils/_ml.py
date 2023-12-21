import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from numpy.typing import NDArray


def cross_validate_report(
    X: NDArray,
    y: NDArray,
    model,
    n_splits: int = 5,
    random_state: int = 3438,
):
    """
    1 - Take a model, and dataset
    2 - Use k-fold cross validation to compute the classification report, confusion matrix and test set classification
           error for each fold. The average of these values is returned.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracy_scores = np.array([])

    precision_scores = np.empty((0, 5))
    recall_scores = np.empty((0, 5))
    f1_scores = np.empty((0, 5))
    support_list = np.empty((0, 3))

    # confusion matrix
    cmatrix = np.zeros((3, 3))

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_test, y_pred = (
            y[test_index],
            model.fit(X_train, y[train_index]).predict(X_test),
        )

        cmatrix += confusion_matrix(y_test, y_pred, labels=[1, 2, 3], normalize='all')

        precision, recall, f1_score, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )

        weighted_avg = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )[0:3]
        macro_avg = precision_recall_fscore_support(y_test, y_pred, average='macro')[
            0:3
        ]

        accuracy_scores = np.append(accuracy_scores, accuracy_score(y_test, y_pred))

        precision_scores = np.vstack(
            [
                precision_scores,
                np.concatenate([precision, [macro_avg[0]], [weighted_avg[0]]], axis=0),
            ]
        )

        recall_scores = np.vstack(
            [
                recall_scores,
                np.concatenate([recall, [macro_avg[1]], [weighted_avg[1]]], axis=0),
            ]
        )

        f1_scores = np.vstack(
            [
                f1_scores,
                np.concatenate([f1_score, [macro_avg[2]], [weighted_avg[2]]], axis=0),
            ]
        )

        support_list = np.vstack([support_list, support])

    report = pd.DataFrame(
        {
            'Precision': precision_scores.mean(axis=0),
            'Recall': recall_scores.mean(axis=0),
            'F1-score': f1_scores.mean(axis=0),
            'True count': np.concatenate(
                [np.round(support_list.mean(axis=0), 0), [0, 0]]
            ),
        },
        index=['1', '2', '3', 'Macro Avg', 'Weighted Avg'],
    )

    cmatrix = pd.DataFrame(
        cmatrix / n_splits, index=['1', '2', '3'], columns=['1', '2', '3']
    )
    cmatrix['Total'] = cmatrix.sum(axis=1)
    cmatrix.loc['Total'] = cmatrix.sum(axis=0)

    test_set_classification_error = 1 - accuracy_scores.mean()

    return report, cmatrix, test_set_classification_error


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


def identify_outliers(X: pd.DataFrame | np.ndarray, z_threshold=3.0):
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
