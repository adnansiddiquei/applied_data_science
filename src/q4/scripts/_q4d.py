from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from src.utils import compute_confidence_interval, format_axes, save_fig


def compute_num_cores_to_utilise():
    num_cores = cpu_count()

    if num_cores > 8:
        return 8
    elif num_cores > 2:
        return num_cores - 2
    else:
        return 1


def compute_oob_wrt_n_estimators(X, y, random_seed: int) -> list[float]:
    np.random.seed(random_seed)

    n_estimators = list(range(5, 501, 5))

    oob_error_rate = []

    for n_estimator in n_estimators:
        clf = RandomForestClassifier(n_estimators=n_estimator, oob_score=True)
        clf.fit(X, y)

        oob_error_rate.append(1 - clf.oob_score_)

    return oob_error_rate


def q4d():
    cwd = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        f'{cwd}/../outputs/ADS_baselineDataset_preprocessed.csv', index_col=0
    )

    data, classifications = data[data.columns[:-1]], data['type']

    X = data.copy().values
    y = classifications.copy().values

    all_args = [(X, y, seed) for seed in range(20)]

    with Pool(compute_num_cores_to_utilise()) as pool:
        results = pool.starmap(
            compute_oob_wrt_n_estimators, [args for args in all_args]
        )

    results = np.column_stack(results)
    n_estimators = list(range(5, 501, 5))

    fig, ax = plt.subplots(figsize=(10, 5))

    means = np.mean(results, axis=1)
    errors = compute_confidence_interval(np.std(results, axis=1))

    plt.errorbar(n_estimators, means, yerr=errors, capsize=2)

    idx = 39
    plt.axhline(y=means[idx], color='gray', linestyle='--')
    plt.axvline(x=n_estimators[idx], color='gray', linestyle='--')

    plt.xlabel('Number of trees')
    plt.ylabel('OOB error rate')

    format_axes(ax)
    save_fig(__file__, 'q4d.png')
