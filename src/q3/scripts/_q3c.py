from sklearn.impute import KNNImputer, SimpleImputer
from src.utils import load_dataset, load_dict_from_json, save_fig
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os


def q3c():
    # Load dataset
    data = load_dataset('C_MissingFeatures.csv', ['Unnamed: 0'])
    data, classifications = data[data.columns[:-1]], data['classification']

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Load the results from q3a, the columns and rows with NaN values
    q3a_results = load_dict_from_json(__file__, 'q3a.json')
    rows_with_nan, columns_with_nan = (
        q3a_results['rows_with_nan'],
        q3a_results['columns_with_nan'],
    )

    # Impute the missing values using KNNImputer, with optimal value computed in q3c_optimise_knn_imputer.py
    knn_imputer = KNNImputer(n_neighbors=15)

    # Also impute the data with a static mean imputation for comparison
    mean_imputer = SimpleImputer(strategy='mean')

    # This will store "data" with the missing values imputed
    knn_imputed_data = pd.DataFrame(columns=data.columns)
    mean_imputed_data = pd.DataFrame(columns=data.columns)

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

        # Impute the missing values with mean
        mean_imputed_data_in_classification = pd.DataFrame(
            mean_imputer.fit_transform(data_in_classification),
            index=data_in_classification.index,
            columns=data_in_classification.columns,
        )

        knn_imputed_data = pd.concat(
            [knn_imputed_data, imputed_data_in_classification], axis=0
        )
        mean_imputed_data = pd.concat(
            [mean_imputed_data, mean_imputed_data_in_classification], axis=0
        )

    knn_imputed_data = knn_imputed_data.sort_index()
    mean_imputed_data = mean_imputed_data.sort_index()

    # Invert the standardisation
    data = pd.DataFrame(scaler.inverse_transform(data), columns=data.columns)
    knn_imputed_data = pd.DataFrame(
        scaler.inverse_transform(knn_imputed_data), columns=data.columns
    )
    mean_imputed_data = pd.DataFrame(
        scaler.inverse_transform(mean_imputed_data), columns=data.columns
    )

    knn_imputed_data['classification'] = classifications
    mean_imputed_data['classification'] = classifications

    affected_data = knn_imputed_data.loc[rows_with_nan][
        columns_with_nan + ['classification']
    ]

    # Note that the sample numbers differ to row index numbers, because sample 1 refers to row 0
    affected_data.index = affected_data.index + 1

    # Plot the table of imputed NaN values
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')

    tbl = pd.plotting.table(
        ax, affected_data.round(2), loc='center', cellLoc='center', rowLoc='center'
    )

    save_fig(__file__, 'q3c_1.png')

    # Now we want to analyse how this imputation has affected the data
    analysis = pd.DataFrame(index=columns_with_nan)
    analysis['Var(KNN) / Var(orig) %'] = 0.0
    analysis['Var(mean) / Var(orig) %'] = 0.0
    analysis['Var(KNN) / Var(mean) %'] = 0.0
    analysis['KS(orig, KNN)'] = 0.0

    for c in columns_with_nan:
        orig_data_var = np.var(data[c])
        knn_imputed_data_var = np.var(knn_imputed_data[c])
        mean_imputed_data_var = np.var(mean_imputed_data[c])

        analysis.loc[c, 'Var(KNN) / Var(orig) %'] = (
            knn_imputed_data_var / orig_data_var
        ) * 100
        analysis.loc[c, 'Var(mean) / Var(orig) %'] = (
            mean_imputed_data_var / orig_data_var
        ) * 100
        analysis.loc[c, 'Var(KNN) / Var(mean) %'] = (
            knn_imputed_data_var / mean_imputed_data_var
        ) * 100

        analysis.loc[c, 'KS(orig, KNN)'] = stats.ks_2samp(
            data[c], knn_imputed_data[c]
        ).pvalue

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    tbl = pd.plotting.table(
        ax, analysis.round(2), loc='center', cellLoc='center', rowLoc='center'
    )

    # Update the fontsize of the header cells
    for i, col in enumerate(analysis.columns):
        tbl[(0, i)].set_height(0.09)

    # Save outputs
    save_fig(__file__, 'q3c_2.png')

    cwd = os.path.dirname(os.path.realpath(__file__))
    knn_imputed_data.round(9).to_csv(
        os.path.join(cwd, '../outputs/knn_imputed_data.csv')
    )
