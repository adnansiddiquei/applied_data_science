import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from src.utils import (
    save_fig,
    cross_validate_report,
    format_contingency_table,
    create_table,
    plot_feature_importance,
    compute_most_important_features_random_forest,
)


def q4e():
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        f'{cwd}/../outputs/ADS_baselineDataset_preprocessed.csv', index_col=0
    )

    data, classifications = data[data.columns[:-1]], data['type']

    X = data.copy().values
    y = classifications.copy().values

    (
        feature_importance,
        most_importance_features,
    ) = compute_most_important_features_random_forest(X, y)

    plot_feature_importance(feature_importance, most_importance_features)
    plt.ylabel('Cumulative Gini Importance')
    save_fig(__file__, 'q4e_gini_importance.png')

    X_subset = data[most_importance_features['feature']].copy().values

    report, cmatrix, test_set_classification_error = cross_validate_report(
        X_subset, y, RandomForestClassifier(random_state=545, n_estimators=200)
    )

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

    save_fig(__file__, 'q4e_confusion_matrix.png')

    tbl = create_table(report.round(3), figsize=(4.5, 2), fontsize=16, scale=(1.6, 2))

    tbl[4, 3].set_text_props(color='white')
    tbl[5, 3].set_text_props(color='white')

    save_fig(__file__, 'q4e_classification_report.png')
