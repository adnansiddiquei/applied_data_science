from src.utils import (
    AdjustedKNNImputer,
    load_dataset,
    create_output_dir_if_required,
    save_dataframe_to_csv,
)
import pandas as pd
import os


def q4b():
    cwd = os.path.dirname(os.path.realpath(__file__))
    create_output_dir_if_required(__file__)

    # Load the data
    data = load_dataset('ADS_baselineDataset.csv', ['Unnamed: 0'])
    data, classifications = data[data.columns[:-1]], data['type']

    # Only imputing outliers, there are no duplicates or missing values
    outliers_imputed = pd.DataFrame(
        AdjustedKNNImputer().fit_transform(data), columns=data.columns
    )

    outliers_imputed['type'] = classifications

    save_dataframe_to_csv(
        outliers_imputed, __file__, 'ADS_baselineDataset_preprocessed.csv'
    )
