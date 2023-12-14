from .q1utils import kmeans_on_dataset_a
from src.utils import save_fig

import warnings

warnings.filterwarnings('ignore')


def q1c():
    # Compute k-means for k = 8 (default value), then plot the confusion matrix, see function for more details
    kmeans_on_dataset_a(n_clusters=8, random_state=3438)
    save_fig(__file__, 'q1c.png', dpi=500)
