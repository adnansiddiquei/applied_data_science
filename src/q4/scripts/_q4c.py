import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from src.utils import format_contingency_table, save_fig, create_table


def q4c():
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        f'{cwd}/../outputs/ADS_baselineDataset_preprocessed.csv', index_col=0
    )
    data, classifications = data[data.columns[:-1]], data['type']

    X = data.copy().values
    y = classifications.copy().values

    # The `stratify=y` gives y_test.value_counts() of {1: 41, 2: 35, 3: 24}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3438, stratify=y
    )

    clf = RandomForestClassifier(random_state=3438)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Plot the confusion matrix
    cmatrix = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
    cmatrix = np.column_stack((cmatrix, cmatrix.sum(axis=1)))
    cmatrix = np.row_stack((cmatrix, cmatrix.sum(axis=0)))

    tbl = format_contingency_table(
        cmatrix,
        columns=['1', '2', '3', 'Total (actual)'],
        index=['1', '2', '3', 'Total (predictions)'],
        figsize=(5, 2),
    )

    tbl[4, 3].set_facecolor('white')
    tbl[4, 3].set_text_props(color='white')

    save_fig(__file__, 'q4c_confusion_matrix.png')

    # Plot the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose().round(2)
    report['support'] = report['support'].astype(int)
    report = report.drop(index='accuracy')

    report = report.rename(columns={'support': 'true count'})

    def capitalise(strings: list[str]) -> list[str]:
        return [string.capitalize() for string in strings]

    report.columns = capitalise(report.columns)
    report.index = capitalise(report.index)

    tbl = create_table(report, figsize=(5, 2))

    tbl[4, 3].set_text_props(color='white')
    tbl[5, 3].set_text_props(color='white')

    save_fig(__file__, 'q4c_classification_report.png')
