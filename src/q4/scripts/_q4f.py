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
    format_axes,
    compute_most_important_features_logit,
    compute_rolling_intersection_pct,
)
import numpy as np
import matplotlib.pyplot as plt
from ._q4e import compute_most_important_features_random_forest


def plot_feature_importance_with_rolling_overlap(X, y):
    X = X.copy()
    y = y.copy()

    (
        feature_importance_logit,
        most_importance_features_logit,
    ) = compute_most_important_features_logit(X, y)
    (
        feature_importance_rf,
        most_importance_features_rf,
    ) = compute_most_important_features_random_forest(X, y)

    rolling_overlap = (
        compute_rolling_intersection_pct(
            feature_importance_logit['feature'].values,
            feature_importance_rf['feature'].values,
        )
        * 100
    )

    fig, ax = plot_feature_importance(
        feature_importance_logit, most_importance_features_logit, label='NMSLI'
    )
    plt.ylabel('Cum. Norm. Mean Squared Logit Importance (NMSLI)')

    ax2 = ax.twinx()
    ax2.plot(rolling_overlap, label='Rolling Feature Overlap %', color='darkorange')
    ax2.set_ylabel('Rolling Feature Overlap %')

    format_axes([ax, ax2], combine_legends=True)

    crossover = rolling_overlap[len(most_importance_features_logit)]

    plt.axhline(crossover, color='grey', linestyle='--')

    plt.text(
        980,
        crossover - 2,
        f'{crossover:.1f}%',
        ha='right',
        va='top',
        fontsize=12,
        color='grey',
    )

    ax.autoscale(enable=True, tight=True, axis='x')

    return fig, ax


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

    # Now compute the most important features
    (
        feature_importance,
        most_importance_features,
    ) = compute_most_important_features_logit(X, y)

    fig, ax = plot_feature_importance_with_rolling_overlap(X, y)
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
