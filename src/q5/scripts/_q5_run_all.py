from . import q5a, q5b, q5c
import time


def q5_run_all():
    print('Running all q5 scripts.')
    start = time.time()
    q5a()
    q5b()
    q5c()
    end = time.time()
    print(f'All q5 scripts finished in {end-start:.0f} seconds.')
    print('---------------------------------------')
