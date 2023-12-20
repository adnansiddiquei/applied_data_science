import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from src.utils import load_dataset, format_axes, save_fig
import matplotlib.pyplot as plt


def set_random_nans(arr, percent=0.01, seed=3438):
    """Randomly sets a given percentage of values in 2d array to NaN."""
    arr = arr.copy()
    np.random.seed(seed)

    num_cells_to_nan = int(arr.size * percent)

    rows = np.random.randint(0, arr.shape[0], size=num_cells_to_nan)
    cols = np.random.randint(0, arr.shape[1], size=num_cells_to_nan)

    for row, col in zip(rows, cols):
        arr[row, col] = np.nan

    return arr


def optimise_knn_imputer(y: np.ndarray, n_neighbors: list):
    """Optimises the n_neighbors parameter for KNNImputer."""
    y = y.copy()

    results = pd.DataFrame(
        columns=['n_neighbors', 'mse', 'mse_std', 'variance', 'variance_std']
    )

    for n in n_neighbors:
        results_for_n = pd.DataFrame(columns=['n_neighbors', 'mse', 'variance'])

        for i in range(10):
            X = set_random_nans(y, 0.05, seed=i)
            imputed_X = KNNImputer(n_neighbors=n).fit_transform(X)

            results_for_n.loc[i] = [
                n,
                mean_squared_error(imputed_X, y),
                np.var(imputed_X) / np.var(y),
            ]

        results.loc[n] = [
            n,
            results_for_n['mse'].mean(),
            results_for_n['mse'].std(),
            results_for_n['variance'].mean(),
            results_for_n['variance'].std(),
        ]

    results = results.reset_index(drop=True)

    return results


def q3c_optimise_knn_imputer():
    """Optimises the n_neighbors parameter for KNNImputer."""
    # Load dataset
    data = load_dataset('C_MissingFeatures.csv', ['Unnamed: 0']).dropna()
    data, classifications = data[data.columns[:-1]], data['classification']

    Ns = [2, 5, 10, 15, 20, 30]

    results = optimise_knn_imputer(data.values, Ns)

    # Plot the results
    fig, ax = plt.subplots()
    plt.xlabel('n_neighbors')

    plt.errorbar(
        x=results['n_neighbors'],
        y=results['variance'],
        yerr=results['variance_std'],
        marker='o',
        capsize=2,
        label='Variance',
    )

    ax2 = ax.twinx()

    plt.errorbar(
        x=results['n_neighbors'],
        y=results['mse'],
        yerr=results['mse_std'],
        marker='o',
        capsize=2,
        c='orange',
        label='MSE',
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
    ax2.set_ylabel('MSE')

    save_fig(__file__, 'q3c_optimise_knn_imputer.png')
