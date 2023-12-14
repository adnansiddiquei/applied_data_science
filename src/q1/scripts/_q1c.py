from .q1utils import kmeans_on_dataset_a
import matplotlib.pyplot as plt
import os

import warnings

warnings.filterwarnings('ignore')


def q1c():
    # Compute k-means for k = 8 (default value), then plot the confusion matrix, see function for more details
    kmeans_on_dataset_a(n_clusters=8, random_state=3438)

    cwd = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(
        os.path.join(cwd, '../outputs/q1c.png'),
        bbox_inches='tight',
        dpi=500,
    )
