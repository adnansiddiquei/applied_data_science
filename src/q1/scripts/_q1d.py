import matplotlib.pyplot as plt
import os
from .q1utils import kmeans_on_dataset_a


def q1d():
    # Re-do question 1c with k = 3
    kmeans_on_dataset_a(n_clusters=3, random_state=3438)

    cwd = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(
        os.path.join(cwd, '../outputs/q1d.png'),
        bbox_inches='tight',
        dpi=500,
    )
