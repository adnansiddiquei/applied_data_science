from . import q1a, q1b, q1c, q1d, q1e
import time


def q1_run_all():
    print('Running all q1 scripts.')
    start = time.time()
    q1a()
    q1b()
    q1c()
    q1d()
    q1e()
    end = time.time()
    print(f'All q1 scripts finished in {end-start:.0f} seconds.')
    print('----------------------------')
