import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from src.utils import (
    format_contingency_table,
    save_fig,
    create_table,
    cross_validate_report,
    load_dataframe_from_csv,
)


def q4c():
    cwd = os.path.dirname(os.path.realpath(__file__))

    # Load data
    data = load_dataframe_from_csv(
        __file__, 'ADS_baselineDataset_preprocessed.csv', index_col=0
    )

    data, classifications = data[data.columns[:-1]], data['type']

    X = data.copy().values
    y = classifications.copy().values

    # Create a confusion matrix and classification report for the RandomForestClassifier
    report, cmatrix, test_set_classification_error = cross_validate_report(
        X, y, RandomForestClassifier(random_state=3438), n_splits=5
    )

    # Output the confusion matrix and classification report
    tbl = format_contingency_table(
        np.round(cmatrix.values, 3),
        columns=['1', '2', '3', 'Tot. (actual)'],
        index=['1', '2', '3', 'Tot. (predictions)'],
        figsize=(4.5, 2),
        fontsize=16,
        scale=(1.6, 2),
    )

    tbl[4, 3].set_facecolor('white')
    tbl[4, 3].set_text_props(color='white')

    save_fig(__file__, 'q4c_confusion_matrix.png')

    tbl = create_table(report.round(3), figsize=(4.5, 2), fontsize=16, scale=(1.6, 2))

    tbl[4, 3].set_text_props(color='white')
    tbl[5, 3].set_text_props(color='white')

    save_fig(__file__, 'q4c_classification_report.png')
