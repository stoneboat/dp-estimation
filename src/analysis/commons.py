import logging
import multiprocessing

import numpy as np

from utils.constants import WORKERS
from utils.empirical_bootstrap import EmpiricalBootstrap, SampleGenerator


def estimator_process_wrapper(kwargs):
    estimator_cls = kwargs['estimator_cls']
    config = kwargs['config']
    estimator = estimator_cls(config)
    return estimator.build()


def batch_estimator_total_report(kwargs_lists, workers):
    pool = multiprocessing.Pool(processes=workers)
    results_list = pool.map(estimator_process_wrapper, kwargs_lists)
    return results_list


def prune_pos_samples(samples, threshold=1000, dim=1):
    if dim == 1:
        index = np.argwhere(np.abs(samples) < threshold)
        index = index.transpose()[0]
        return samples[index]
    else:
        index = np.argwhere(np.linalg.norm(samples, ord=np.inf, axis=1) < threshold)
        index = index.transpose()[0]
        return samples[index]


def batch_estimator_estimated_delta(kwargs_lists, workers):
    pool = multiprocessing.Pool(processes=workers)
    results_list = pool.map(estimator_process_wrapper, kwargs_lists)

    estimated_delta = np.zeros(len(kwargs_lists))
    for i in range(len(kwargs_lists)):
        estimated_delta[i] = results_list[i]['estimated_delta']

    return estimated_delta


def compute_bootstrap_range(estimator_cls, config, n_samples, workers=WORKERS, confidence_interval_prob=0.9,
                            bootstrap_samples=10):
    kwargs = {'estimator_cls': estimator_cls, 'config': config}
    input_list = []
    for i in range(n_samples):
        input_list.append(kwargs)

    pool = multiprocessing.Pool(processes=workers)

    results_list = pool.map(estimator_process_wrapper, input_list)

    estimated_delta = np.zeros(n_samples)
    for i in range(n_samples):
        estimated_delta[i] = results_list[i]['estimated_delta']

    bootstrap = EmpiricalBootstrap(sample_generator=SampleGenerator(data=estimated_delta))

    boot_res = bootstrap.bootstrap_confidence_bounds(
        confidence_interval_prob=confidence_interval_prob,
        n_samples=bootstrap_samples
    )
    logging.critical(boot_res)
    return boot_res

