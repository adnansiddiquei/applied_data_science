import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.utils import (
    cross_validate_report,
    format_contingency_table,
    create_table,
    save_fig,
    plot_feature_importance,
)
import numpy as np
import matplotlib.pyplot as plt


def compute_most_important_features_logit(X: np.ndarray, y: np.ndarray):
    X = X.copy()
    y = y.copy()

    pipeline = Pipeline(
        [
            ('scaler', StandardScaler()),
            (
                'logistic_regression',
                LogisticRegression(multi_class='multinomial', random_state=3438),
            ),
        ]
    )

    pipeline.fit(X, y)

    feature_importance = (
        pd.DataFrame(pipeline['logistic_regression'].coef_.T ** 2)
        .mean(axis=1)
        .reset_index()
        .sort_values(0, ascending=False)
        .reset_index(drop=True)
        .rename(columns={'index': 'feature', 0: 'importance'})
    )

    # Normalise the feature importance
    feature_importance['importance'] = (
        feature_importance['importance'] / feature_importance['importance'].sum()
    )

    feature_importance['feature'] = [
        f'Fea{feature + 1}' for feature in feature_importance['feature']
    ]

    feature_importance['cumulative_importance'] = feature_importance[
        'importance'
    ].cumsum()

    most_importance_features = feature_importance[
        feature_importance['cumulative_importance'].round(4) <= 0.95
    ]

    return feature_importance, most_importance_features


def q4f():
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        f'{cwd}/../outputs/ADS_baselineDataset_preprocessed.csv', index_col=0
    )

    data, classifications = data[data.columns[:-1]], data['type']

    X = data.copy().values
    y = classifications.copy().values

    # First, we'll try with all features
    pipeline = Pipeline(
        [
            ('scaler', StandardScaler()),
            (
                'logistic_regression',
                LogisticRegression(multi_class='multinomial', random_state=3438),
            ),
        ]
    )

    report, cmatrix, test_set_classification_error = cross_validate_report(
        X, y, pipeline, n_splits=5
    )

    tbl = format_contingency_table(
        np.round(cmatrix.values, 2),
        columns=['1', '2', '3', 'Total (actual)'],
        index=['1', '2', '3', 'Total (predictions)'],
        figsize=(5, 2),
    )

    tbl[4, 3].set_facecolor('white')
    tbl[4, 3].set_text_props(color='white')

    save_fig(__file__, 'q4f_confusion_matrix_all_feats.png')

    tbl = create_table(report.round(2), figsize=(5, 2))

    tbl[4, 3].set_text_props(color='white')
    tbl[5, 3].set_text_props(color='white')

    save_fig(__file__, 'q4f_classification_report_all_feats.png')

    # Now compute the most important features]
    (
        feature_importance,
        most_importance_features,
    ) = compute_most_important_features_logit(X, y)

    plot_feature_importance(feature_importance, most_importance_features)
    plt.ylabel('Cum. Mean Squared Logit Importance (MSLI)')
    save_fig(__file__, 'q4f_logit_importance.png')

    X_subset = data[most_importance_features['feature']].copy().values

    report, cmatrix, test_set_classification_error = cross_validate_report(
        X_subset, y, pipeline, n_splits=5
    )

    tbl = format_contingency_table(
        np.round(cmatrix.values, 2),
        columns=['1', '2', '3', 'Total (actual)'],
        index=['1', '2', '3', 'Total (predictions)'],
        figsize=(5, 2),
    )

    tbl[4, 3].set_facecolor('white')
    tbl[4, 3].set_text_props(color='white')

    save_fig(__file__, 'q4f_confusion_matrix_reduced_feats.png')

    tbl = create_table(report.round(2), figsize=(5, 2))

    tbl[4, 3].set_text_props(color='white')
    tbl[5, 3].set_text_props(color='white')

    save_fig(__file__, 'q4f_classification_report_reduced_feats.png')
