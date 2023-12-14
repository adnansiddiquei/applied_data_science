from .q1utils import kmeans_on_dataset_a
from src.utils import save_fig


def q1d():
    # Re-do question 1c with k = 3
    kmeans_on_dataset_a(n_clusters=3, random_state=3438)
    save_fig(__file__, 'q1d.png', dpi=500)
