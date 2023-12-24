from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from .ml import identify_outliers
from typing import Literal
from sklearn.impute import SimpleImputer


class AdjustedKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        z_score_threshold=3,
        n_neighbors=15,
        n_components=2,
        iter_max=None,
        impute_type: Literal['outliers', 'nans'] = 'outliers',
        random_state=3438,
    ):
        self.z_score_threshold = z_score_threshold
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.iter_max = iter_max
        self.impute_type = impute_type
        self.random_state = random_state

        self.X: np.ndarray | None = None
        self.nan_positions: np.ndarray | None = None
        self.new_outliers: np.ndarray | None = None

        self.n_iters = 0

    def fit(self, X: pd.DataFrame | np.ndarray, y=None):
        np.random.seed(self.random_state)

        if self.impute_type == 'nans':
            self.nan_positions = (
                np.isnan(X).values if isinstance(X, pd.DataFrame) else np.isnan(X)
            )

        return self

    def _identify_outliers(self, X):
        if self.impute_type == 'outliers':
            return identify_outliers(X, z_threshold=self.z_score_threshold)
        elif self.impute_type == 'nans':
            return self.nan_positions
        else:
            raise ValueError(f'Invalid impute_type: {self.impute_type}')

    def _one_passthrough(self, X):
        scaler = StandardScaler()

        # Standardize the data
        X = X.copy().values if isinstance(X, pd.DataFrame) else X.copy()
        X_scaled = scaler.fit_transform(X)

        # Identify outliers
        outliers = self._identify_outliers(X)

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

        new_outliers = self._identify_outliers(X)

        return X, new_outliers

    def transform(self, X: pd.DataFrame | np.ndarray, y=None):
        """Impute outliers until it gets down to the expected number of outliers."""
        np.random.seed(self.random_state)

        self.X = X.copy().values if isinstance(X, pd.DataFrame) else X.copy()
        X = self.X.copy()

        if self.impute_type == 'outliers':
            expected_num_outliers = (
                1
                - (norm.cdf(self.z_score_threshold) - norm.cdf(-self.z_score_threshold))
            ) * np.prod(X.shape)

            expected_num_outliers *= 1.05  # Add a 5% buffer

            X_new, outliers = self._one_passthrough(X)

            self.n_iters = 1

            def iter_condition():
                if self.iter_max is not None:
                    return self.n_iters < self.iter_max
                else:
                    return True

            # Iteratively impute outliers until we get down to the expected number of outliers
            while (
                self._identify_outliers(
                    X_new,
                ).sum()
                > expected_num_outliers
                and iter_condition()
            ):
                X_new, outliers = self._one_passthrough(X_new)
                self.n_iters += 1

            self.new_outliers = outliers

            return X_new
        elif self.impute_type == 'nans':
            # If we are imputing nans, then for the moment, replace nans with the mean
            X = SimpleImputer(strategy='mean').fit_transform(X)
            return self._one_passthrough(X)[0]
        else:
            raise ValueError(f'Invalid impute_type: {self.impute_type}')
