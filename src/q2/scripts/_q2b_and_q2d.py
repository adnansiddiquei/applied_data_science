from src.utils import load_dataset, save_fig
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np


def q2b_and_q2d():
    data = load_dataset('B_Relabelled.csv', ['Unnamed: 0'], standardise=True)
    data, classifications = data[data.columns[:-1]], data['classification']

    # Get all duplicates
    duplicates = data[data.duplicated(keep=False)]

    # This is a dataframe where column 0 gives us a feature, column 1 gives us a duplicate of that feature
    duplicate_mapping = pd.DataFrame(
        duplicates.groupby(list(duplicates.columns))
        .apply(lambda x: tuple(x.index))
        .tolist()
    )

    # This is now a dataframe where column 0 gives us a feature, column 1 gives us a duplicate of that feature
    # column 3 (called 'c0') gives us the classification of the feature in column 0, and column 4 (called 'c1')
    # gives us the classification of the feature in column 1
    duplicate_mapping = (
        duplicate_mapping.merge(classifications, left_on=0, right_index=True)
        .rename({'classification': 'c0'}, axis=1)
        .merge(classifications, left_on=1, right_index=True)
        .rename({'classification': 'c1'}, axis=1)
    )

    duplicate_mapping['c0'] = duplicate_mapping['c0'].astype(int)
    duplicate_mapping['c1'] = duplicate_mapping['c1'].astype(int)

    # This gets rif of all the rows in duplicate_mapping where the duplicated features have been assigned the same
    # classification
    dm_mislabelled = duplicate_mapping[
        duplicate_mapping['c0'] != duplicate_mapping['c1']
    ]

    # This now trains a Logistic regression model on all of the non-duplicated features with non-missing labels
    model1 = LogisticRegression(multi_class='multinomial', random_state=3428)

    y = classifications[~data.duplicated(keep=False)].dropna()
    X = data.loc[y.index]

    model1.fit(X, y)

    # This now predicts the classification of the mislabelled features
    correct_labels = model1.predict(data.loc[dm_mislabelled[0]])

    # This now replaces the mislabelled features with the correct labels
    new_classifications = classifications.copy()

    # The number that exited each classification as they were found to be mislabelled
    tracker = (
        -new_classifications.loc[dm_mislabelled[0]].value_counts().sort_index().values
    )

    new_classifications.loc[dm_mislabelled[0]] = correct_labels

    # The number that entered each classification after correct labelling
    tracker = np.column_stack(
        [tracker, pd.Series(correct_labels).value_counts().sort_index().values]
    )

    # Now remove one version of the duplicated features from the dataset
    new_classifications = new_classifications.drop(duplicate_mapping[1])
    new_data = data.drop(duplicate_mapping[1])

    # The number that exited each classification after dropping duplicates
    tracker = np.column_stack(
        [tracker, -duplicate_mapping['c1'].value_counts().sort_index().values]
    )

    # Now train a new model on the new dataset which has no duplicates
    model2 = LogisticRegression(multi_class='multinomial', random_state=3428)

    y = new_classifications.dropna()
    X = new_data.loc[y.index]

    model2.fit(X, y)

    missing_labels = model2.predict(new_data[new_classifications.isna()])

    new_classifications[new_classifications.isna()] = missing_labels

    # The number that entered each classification after imputing missing labels
    tracker = np.column_stack(
        [tracker, pd.Series(missing_labels).value_counts().sort_index().values]
    )

    # Now to save the new summary table
    # Get the counts of each classification
    counts = pd.DataFrame(new_classifications.value_counts())
    counts.loc[5] = 0
    counts.loc[6] = counts.sum()
    counts['count'] = counts['count'].astype(str)
    counts.index = ['1', '2', '4', 'Missing', 'Total']

    for i, index in enumerate(['1', '2', '4']):
        counts.loc[index] = (
            f'{counts["count"].loc[index]}'
            f'    {tracker[i][0], tracker[i][1], tracker[i][2], tracker[i][3]}'
        )

    counts.index = ['    ' + index + '    ' for index in counts.index]
    counts.columns = ['Count']

    # Plot the table
    fig, ax = plt.subplots(figsize=(2, 2))  # set size frame
    ax.axis('off')
    tbl = pd.plotting.table(ax, counts, loc='center', cellLoc='center', rowLoc='center')

    # Format the table
    tbl[(5, -1)].set_text_props(weight='bold')
    tbl[(5, 0)].set_text_props(weight='bold')

    # Highlight the cells based on their value with a color map
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(2, 1.5)

    save_fig(__file__, 'q2b_q2d.png')
