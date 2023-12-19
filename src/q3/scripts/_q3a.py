from src.utils import load_dataset, save_fig, save_dict_to_json
import matplotlib.pyplot as plt
import pandas as pd


def q3a():
    # Load dataset
    data = load_dataset('C_MissingFeatures.csv', ['Unnamed: 0'])
    data, classifications = data[data.columns[:-1]], data['classification']

    # Find the columns (features) and rows (samples) with NaN values
    columns_with_nan = list(data.columns[data.isna().any()])
    rows_with_nan = data.drop(data.dropna().index).index

    # Get the data that has NaN values
    affected_data = data.loc[rows_with_nan][columns_with_nan]

    # Note that the sample numbers differ to row index numbers, because sample 1 refers to row 0
    affected_data.index = affected_data.index + 1

    # Plot the table of NaN values
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')

    tbl = pd.plotting.table(
        ax, affected_data, loc='center', cellLoc='center', rowLoc='center'
    )

    # Save the results from this question to a JSON file, so it can be loaded in to q3c
    save_dict_to_json(
        {
            'rows_with_nan': list(rows_with_nan),
            'columns_with_nan': columns_with_nan,
            'samples_with_nan': list(rows_with_nan + 1),
        },
        __file__,
        'q3a.json',
    )

    save_fig(__file__, 'q3a.png')
