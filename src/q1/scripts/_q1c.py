from .q1utils import kmeans_on_dataset_a
from src.utils import save_fig

import warnings

warnings.filterwarnings('ignore')


def q1c():
    """Q1c

    Partition the data into two training sets of equal size. Apply k-means clustering to each training set, using
    the default sci-kit-learn parameters. In each case, the unused data can be mapped onto the learned clusters.

    Compare both clusterings for the combined training set, using a contingency table.
    """
    # Compute k-means for k = 8 (default value), then plot the confusion matrix, see function for more details
    kmeans_on_dataset_a(n_clusters=8, random_state=3438)
    save_fig(__file__, 'q1c.png', dpi=500)
