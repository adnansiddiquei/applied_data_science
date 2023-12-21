from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm


class AdjustedKNNOutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self, z_score_threshold=3, n_neighbors=15, n_components=2, iter_max=None
    ):
        self.z_score_threshold = z_score_threshold
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.iter_max = iter_max
        self.standard_scaler = StandardScaler()

        self.X: np.ndarray | None = None
        self.X_scaled: np.ndarray | None = None
        self.outliers: np.ndarray | None = None

        self.new_outliers: np.ndarray | None = None

        self.pca = PCA(n_components=self.n_components)
        self.X_scaled_pca: np.ndarray | None = None

        self.neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1)
        self.nearest_neighbors: np.ndarray | None = None

        self.n_iters = 0

    def fit(self, X: pd.DataFrame | np.ndarray, y=None):
        np.random.seed(3438)
        return self

    def _one_passthrough(self, X):
        scaler = StandardScaler()

        # Standardize the data
        X = X.copy().values if isinstance(X, pd.DataFrame) else X.copy()
        X_scaled = scaler.fit_transform(X)

        # Identify outliers
        outliers = self.identify_outliers(X, z_threshold=self.z_score_threshold)

        # Apply PCA
        pca = PCA(n_components=self.n_components)
        X_scaled_pca = pca.fit_transform(X_scaled)

        # Find the nearest neighbors for every sample
        neighbors = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        neighbors.fit(X_scaled_pca)
        nearest_neighbors = neighbors.kneighbors(X_scaled_pca, return_distance=False)[
            :, 1:
        ]

        # Now with the nearest neighbors, impute the outliers
        for sample, outliers_in_sample, nearest_neighbors_for_sample in zip(
            X, outliers, nearest_neighbors
        ):
            if len(outliers_in_sample) == 0:
                continue

            # compute the impute values:
            #   1. Get all the nearest neighbors for the sample
            #   2. Get the features in the nearest neighbors that the sample contains outliers in
            #   3. Compute the mean of those features
            impute_values = np.mean(
                X[nearest_neighbors_for_sample][:, outliers_in_sample], axis=0
            )

            #   4. Replace the outliers in the sample with the impute values
            sample[outliers_in_sample] = impute_values

        new_outliers = self.identify_outliers(X, z_threshold=self.z_score_threshold)

        return X, new_outliers

    @staticmethod
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

    def transform(self, X: pd.DataFrame | np.ndarray, y=None):
        """Impute outliers until it gets down to the expected number of outliers."""
        self.X = X.copy().values if isinstance(X, pd.DataFrame) else X.copy()

        expected_num_outliers = (
            1 - (norm.cdf(self.z_score_threshold) - norm.cdf(-self.z_score_threshold))
        ) * np.prod(X.shape)

        expected_num_outliers *= 1.05  # Add a 5% buffer

        X = self.X.copy()

        X_new, outliers = self._one_passthrough(X)

        self.n_iters = 1

        def iter_condition():
            if self.iter_max is not None:
                return self.n_iters < self.iter_max
            else:
                return True

        # Iteratively impute outliers until we get down to the expected number of outliers
        while (
            self.identify_outliers(X_new, z_threshold=self.z_score_threshold).sum()
            > expected_num_outliers
            and iter_condition()
        ):
            X_new, outliers = self._one_passthrough(X_new)
            self.n_iters += 1

        self.new_outliers = outliers

        return X_new
