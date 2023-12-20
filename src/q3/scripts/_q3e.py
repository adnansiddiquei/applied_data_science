import os
from .q3utils import knn_impute_outliers, identify_most_discriminative_features
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import format_axes, save_fig


def q3e():
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        os.path.join(cwd, '../outputs/q3c_missing_data_imputed.csv'), index_col=0
    )

    data, classifications = data[data.columns[:-1]], data['classification']

    # There are a total of 2904 outliers, we impute them using KNNImputer and the 38 most discriminative features
    processed_data, new_outliers = knn_impute_outliers(data)

    # Save the processed data
    processed_data.round(9).to_csv(
        os.path.join(cwd, '../outputs/q3e_outliers_imputed.csv')
    )

    # Now we want to analyse how this imputation has affected the data
    # We will see how the mean and variance of the 81 most discriminative features have changed
    discriminative_features = list(
        identify_most_discriminative_features(data, loading_pct_threshold=0.005).index
    )

    # Compute the means and standard deviations of the 81 most discriminative features, for the raw and processed data
    index = [int(idx.strip('Fea')) for idx in discriminative_features]

    mean_raw_data = data.mean(axis=0)[discriminative_features]
    std_raw_data = data.std(axis=0)[discriminative_features]

    summary1 = pd.concat([mean_raw_data, std_raw_data], axis=1)
    summary1.columns = ['mean', 'std']
    summary1.index = index

    mean_processed_data = processed_data.mean(axis=0)[discriminative_features]
    std_processed_data = processed_data.std(axis=0)[discriminative_features]

    summary2 = pd.concat([mean_processed_data, std_processed_data], axis=1)
    summary2.columns = ['mean', 'std']
    summary2.index = index

    # Now we plot this
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.errorbar(
        summary1.index,
        summary1['mean'],
        summary1['std'],
        fmt='o',
        label='Pre-imputation',
        markersize=4,
        alpha=1,
        capsize=6,
        elinewidth=0.4,
    )
    plt.errorbar(
        summary2.index,
        summary2['mean'],
        summary2['std'],
        fmt='x',
        label='Post-imputation',
        markersize=4,
        alpha=0.7,
        capsize=4,
        elinewidth=0.4,
    )

    plt.legend()
    format_axes(ax)
    plt.xlabel('Feature')
    plt.ylabel('Mean')
    save_fig(__file__, 'q3e.png')
