from src.utils import load_dataset, save_fig
import pandas as pd
import matplotlib.pyplot as plt


def q2a():
    # Load the dataset
    data = load_dataset('B_Relabelled.csv', ['Unnamed: 0'])

    data, classifications = data[data.columns[:-1]], data['classification']

    # Get the counts of each classification
    counts = pd.DataFrame(classifications.value_counts(dropna=False))
    counts.loc[-1] = counts.sum()
    counts.index = ['1', '2', '3', 'Missing', 'Total']
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

    save_fig(__file__, 'q2a.png')
