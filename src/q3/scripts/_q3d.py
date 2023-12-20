import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils import format_axes, save_fig
from .q3utils import identify_outliers
import pandas as pd


def q3d():
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        os.path.join(cwd, '../outputs/q3c_missing_data_imputed.csv'), index_col=0
    )
    data, classifications = data[data.columns[:-1]], data['classification']

    # There are a total of 2904 outliers
    outliers = identify_outliers(data)

    # Create the heatmap using seaborn
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(outliers, cmap='viridis', cbar=False)
    plt.xlabel('Features')
    plt.ylabel('Samples')
    format_axes(ax)

    save_fig(__file__, 'q3d_heatmap.png')
