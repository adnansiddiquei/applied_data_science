from . import q2a, q2b_and_q2d
import time


def q2_run_all():
    print('Running all q2 scripts.')
    start = time.time()
    q2a()
    q2b_and_q2d()
    end = time.time()
    print(f'All q2 scripts finished in {end-start:.0f} seconds.')
    print('--------------------------------------')
