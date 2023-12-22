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
)


def compute_most_important_features_random_forest(X: np.ndarray, y: np.ndarray):
    X = X.copy()
    y = y.copy()

    clf = RandomForestClassifier(random_state=3438)

    clf.fit(X, y)

    feature_importance = (
        pd.DataFrame(clf.feature_importances_)
        .reset_index()
        .rename(columns={0: 'gini_importance', 'index': 'feature'})
        .sort_values('gini_importance', ascending=False)
        .reset_index(drop=True)
    )

    feature_importance['feature'] = [
        f'Fea{feature + 1}' for feature in feature_importance['feature']
    ]
    feature_importance['cumulative_importance'] = feature_importance[
        'gini_importance'
    ].cumsum()

    most_importance_features = feature_importance[
        feature_importance['cumulative_importance'].round(4) <= 0.95
    ]

    return feature_importance, most_importance_features


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
        X_subset, y, RandomForestClassifier(random_state=3438)
    )

    tbl = format_contingency_table(
        np.round(cmatrix.values, 2),
        columns=['1', '2', '3', 'Total (actual)'],
        index=['1', '2', '3', 'Total (predictions)'],
        figsize=(5, 2),
    )

    tbl[4, 3].set_facecolor('white')
    tbl[4, 3].set_text_props(color='white')

    save_fig(__file__, 'q4e_confusion_matrix.png')

    tbl = create_table(report.round(2), figsize=(5, 2))

    tbl[4, 3].set_text_props(color='white')
    tbl[5, 3].set_text_props(color='white')

    save_fig(__file__, 'q4e_classification_report.png')
