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
)
import numpy as np
import matplotlib.pyplot as plt
from ._q4e import compute_most_important_features_random_forest


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

    def compute_overlap(fi1, fi2):
        num_features = len(fi1)
        overlap_rolling_pct = np.zeros(num_features - 1)

        for i in range(num_features - 1):
            features_in_fi1 = fi1.iloc[: i + 1]['feature'].values
            features_in_fi2 = fi2.iloc[: i + 1]['feature'].values

            overlap_count = np.intersect1d(features_in_fi1, features_in_fi2)

            overlap_rolling_pct[i] = len(overlap_count) / (i + 1)

        return overlap_rolling_pct

    rolling_overlap = (
        compute_overlap(feature_importance_logit, feature_importance_rf) * 100
    )

    fig, ax = plot_feature_importance(
        feature_importance_logit, most_importance_features_logit, label='MSLI'
    )
    plt.ylabel('Cum. Norm. Mean Squared Logit Importance (NMSLI)')

    ax2 = ax.twinx()
    ax2.plot(rolling_overlap, label='Rolling Feature Overlap %', color='darkorange')
    ax2.set_ylabel('Rolling Feature Overlap %')

    format_axes(ax2, ticks_left=False)
    format_axes(ax, ticks_right=False)

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

    # Collect the legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    handles.extend(handles2)
    labels.extend(labels2)

    # into  a single legend
    ax.legend(handles, labels, loc='lower right')

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
