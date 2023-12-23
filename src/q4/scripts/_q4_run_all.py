from . import q4b, q4c, q4d, q4e, q4f
import time


def q4_run_all():
    print('Running all q4 scripts.')
    start = time.time()
    q4b()
    q4c()
    q4d()
    q4e()
    q4f()
    end = time.time()
    print(f'All q4 scripts finished in {end-start:.0f} seconds.')
    print('---------------------------------------')
