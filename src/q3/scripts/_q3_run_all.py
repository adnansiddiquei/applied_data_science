from . import q3a, q3c, q3c_optimise_knn_imputer, q3d, q3e
import time


def q3_run_all():
    print('Running all q3 scripts.')
    start = time.time()
    q3a()
    q3c()
    q3c_optimise_knn_imputer()
    q3d()
    q3e()
    end = time.time()
    print(f'All q3 scripts finished in {end-start:.0f} seconds.')
    print('--------------------------------------')
