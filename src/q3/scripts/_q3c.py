from sklearn.impute import SimpleImputer
from src.utils import (
    load_dataset,
    load_dict_from_json,
    save_fig,
    AdjustedKNNImputer,
    save_dataframe_to_csv,
)
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

    # Standardise the data
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Load the results from q3a, the columns and rows with NaN values
    q3a_results = load_dict_from_json(__file__, 'q3a.json')
    rows_with_nan, columns_with_nan = (
        q3a_results['rows_with_nan'],
        q3a_results['columns_with_nan'],
    )

    # Use the custom-made AdjustedKNNImputer to impute the data
    knn_imputed_data = pd.DataFrame(
        AdjustedKNNImputer(impute_type='nans').fit_transform(data), columns=data.columns
    )

    # Also impute the data with a static mean imputation for comparison
    mean_imputer = SimpleImputer(strategy='mean')

    mean_imputed_data = pd.DataFrame(
        scaler.inverse_transform(mean_imputer.fit_transform(scaled_data)),
        columns=data.columns,
    )

    # Add the classifications back to the imputed data
    knn_imputed_data['classification'] = classifications
    mean_imputed_data['classification'] = classifications

    # Extract only the samples and feature that originally had missing data, i.e., the cells we just imputed
    affected_data = knn_imputed_data.loc[rows_with_nan][
        columns_with_nan + ['classification']
    ]

    # Note that the sample numbers differ to row index numbers, because sample 1 refers to row 0, correct for this
    # so the outputted table has the index refer to the sample number
    affected_data.index = affected_data.index + 1

    # Plot the table of imputed NaN values
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')

    tbl = pd.plotting.table(
        ax, affected_data.round(2), loc='center', cellLoc='center', rowLoc='center'
    )

    save_fig(__file__, 'q3c_1.png')

    # Now we want to analyse how this imputation has affected the data, compute and place into the analysis dataframe
    # how the variance of the data has changed due to the imputation
    analysis = pd.DataFrame(index=columns_with_nan)
    analysis['Var(KNN) / Var(orig) %'] = ''
    analysis['Var(mean) / Var(orig) %'] = ''
    analysis['Var(KNN) / Var(mean) %'] = ''
    analysis['KS(orig, KNN) p-val'] = ''

    for c in columns_with_nan:
        orig_data_var = np.var(data[c])
        knn_imputed_data_var = np.var(knn_imputed_data[c])
        mean_imputed_data_var = np.var(mean_imputed_data[c])

        analysis.loc[c, 'Var(KNN) / Var(orig) %'] = f'{(
            knn_imputed_data_var / orig_data_var
        ) * 100:.2f}'
        analysis.loc[c, 'Var(mean) / Var(orig) %'] = f'{(
            mean_imputed_data_var / orig_data_var
        ) * 100:.2f}'
        analysis.loc[c, 'Var(KNN) / Var(mean) %'] = f'{(
            knn_imputed_data_var / mean_imputed_data_var
        ) * 100:.2f}'

        analysis.loc[c, 'KS(orig, KNN) p-val'] = f'{stats.ks_2samp(
            data[c], knn_imputed_data[c]
        ).pvalue:.6f}'

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

    # Save the analysis dataframe to a CSV file
    save_dataframe_to_csv(
        knn_imputed_data.round(9), __file__, 'q3c_missing_data_imputed.csv'
    )
