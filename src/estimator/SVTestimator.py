import logging
import os
import time
from enum import Enum

import numpy as np
from scipy.stats import laplace

from mech.SVTMechanism import SVTGenerator, SVTVariant, parallel_build_estimator, DataPattern, \
    gen_test_neighbors, parallel_gen_samples_to_files, parallel_gen_samples
from utils.commons import accuracy_to_delta, compute_tight_bound, create_project_directory
from utils.external_merge import parallel_external_merger


class EstimatorEngine(Enum):
    kNN = 1
    empiricalDistribution = 2


class SVTEstimator:
    def __init__(self, svt_variant, estimator_engine, kwargs):
        assert isinstance(estimator_engine, EstimatorEngine)
        assert isinstance(svt_variant, SVTVariant)
        self.engine = estimator_engine
        self.svt_variant = svt_variant
        self.kwargs = kwargs
        self.training_set_size = kwargs["training_set_size"]
        self.validation_set_size = kwargs["validation_set_size"]
        self.training_batch_size = kwargs['training_batch_size']
        self.testing_batch_size = kwargs['testing_batch_size']
        self.batch_size = kwargs['buffer_size']
        self.gamma = kwargs["gamma"]
        self.dataset_settings = kwargs["dataset_settings"]
        self.sample_generator = SVTGenerator(svt_variant=self.svt_variant, kwargs=self.kwargs)

        self.output_ = None
        self.model = None

        self.tmp_directory = None

    def compute_tight_bound(self, failure_prob):
        def compute_minimum_num_terms_svt1(max_gap, failure_prob, epsilon, claimed_epsilon):
            q = laplace.cdf(max_gap + epsilon/2*np.log(2/failure_prob))
            return 1 + np.ceil(np.log(2/((np.exp(claimed_epsilon)+1)*failure_prob*np.log(1/q)))/np.log(1/q))

        if self.engine == EstimatorEngine.empiricalDistribution:
            if self.svt_variant == SVTVariant.SVT1:
                k = compute_minimum_num_terms_svt1(
                    max_gap=self.sample_generator.max_gap(),
                    failure_prob=failure_prob/2,
                    epsilon=self.dataset_settings['epsilon'],
                    claimed_epsilon=self.dataset_settings['claimed_epsilon']
                )
                if k > self.sample_generator.query_length():
                    logging.warning(f"The query length is smaller than the theoretical recommending query length {k}")
            return np.sqrt(np.power(self.sample_generator.query_length(), 2) /
                           (2 * self.training_set_size) * np.log(4 / failure_prob))

    def build(self, num_workers, in_memory=True):
        if self.engine == EstimatorEngine.empiricalDistribution:
            logging.info('Build empirical distribution estimator')
            self.model = parallel_build_estimator(number_samples=self.training_set_size, svt_variant=self.svt_variant,
                                                  estimator_configure=self.kwargs, batch_size=self.training_batch_size,
                                                  workers=num_workers)

            logging.info('Compute estimated delta')
            accuracy = self.model.compute_accuracy()
            estimated_delta = accuracy_to_delta(accuracy, self.dataset_settings['claimed_epsilon'])
            mu = self.compute_tight_bound(failure_prob=self.gamma)

            estimated_range = (max(0, estimated_delta - mu), max(estimated_delta + mu, 0))
            self.output_ = {
                'estimated_delta': estimated_delta, 'accuracy': accuracy,
                'estimated_range': estimated_range, 'gamma': self.gamma,
                'sample_size': self.training_set_size
            }
            return self.output_

        if self.engine == EstimatorEngine.kNN and in_memory:
            logging.info('Generate and sorted samples')
            tic = time.perf_counter()
            samples = parallel_gen_samples(
                number_samples=self.training_set_size,
                svt_variant=self.svt_variant,
                estimator_configure=self.kwargs, batch_size=self.training_batch_size,
                is_sorted=True, workers=num_workers
            )
            toc = time.perf_counter()
            logging.info(f'Total time cost now: {toc - tic:0.4f} seconds')

            logging.info('Start to compute delta')
            tic = time.perf_counter()
            test_len = self.training_set_size
            tail_p = 0
            head_p = int(np.sqrt(test_len))
            points = samples['X']
            labels = samples['y']
            error_cnt = 0
            index = 0

            while index < test_len:
                value = points[index]
                #   Compute the nearest neighbors
                while head_p < test_len - 1:
                    if np.abs(points[tail_p] - value) >= np.abs(points[head_p] - value):
                        tail_p += 1
                        head_p += 1
                    else:
                        break

                #   Compute its risk and prepare the next item
                error_cnt += np.abs(labels[index] - np.rint(labels[tail_p:head_p].mean()))
                index += 1

                #   Timer
                if index % self.batch_size == 0:
                    toc = time.perf_counter()
                    logging.info(f"computed {index/test_len*100}% in {toc - tic:0.4f} seconds")

            accuracy = 1 - error_cnt/test_len
            toc = time.perf_counter()
            logging.info(f'Total time cost now: {toc - tic:0.4f} seconds')

            estimated_delta = accuracy_to_delta(accuracy, self.dataset_settings['claimed_epsilon'])
            mu = compute_tight_bound(
                gamma=self.gamma, n1=self.training_set_size,
                n2=self.training_set_size, d=len(self.dataset_settings['database_0']),
                epsilon=self.dataset_settings['claimed_epsilon']
            )

            estimated_range = (max(0, estimated_delta - mu), max(estimated_delta + mu, 0))
            self.output_ = {
                'estimated_delta': estimated_delta, 'accuracy': accuracy,
                'estimated_range': estimated_range, 'gamma': self.gamma,
                'sample_size': self.training_set_size
            }
            return self.output_

        if self.engine == EstimatorEngine.kNN and not in_memory:
            logging.info('Generate and sort samples in slice')
            tic = time.perf_counter()
            self.tmp_directory = create_project_directory(directory_name="tmp")
            files_list = parallel_gen_samples_to_files(number_samples=self.training_set_size,
                                                       svt_variant=self.svt_variant,
                                                       estimator_configure=self.kwargs,
                                                       batch_size=self.training_batch_size,
                                                       directory_name=self.tmp_directory,
                                                       is_sorted=True, workers=num_workers)
            toc = time.perf_counter()
            logging.info(f'Total time cost now: {toc - tic:0.4f} seconds')

            logging.info('Merge sorted slice into simple sorted sample files')
            demo_sample = self.sample_generator.gen_samples(num_samples=1, generate_positive_sample=True)
            sorted_file_name = parallel_external_merger(
                sorted_slice_list=files_list, directory_name=self.tmp_directory,
                data_type=demo_sample['X'].dtype, batch_size=self.batch_size,
                num_component=self.sample_generator.dim + 1, less_than_func=self.sample_generator.less_than_func(),
                outfile_name=os.path.join(self.tmp_directory, "bin.out"), num_workers=num_workers
            )
            toc = time.perf_counter()
            logging.info(f'Total time cost now: {toc - tic:0.4f} seconds')


def neighboring_pair_generating_function(query_length, sensitivity=1):
    assert query_length % 2 == 0
    assert sensitivity > 0
    basis = [
        [[0], [-2]], [[1], [-2]],
        [[0], [-1]], [[1], [-1]],
        [[1], [0]]
    ]

    padding_basis_left = [1]
    padding_basis_right = [0]
    half_length = int(query_length/2)
    for length in range(1, half_length+1):
        for index in range(len(basis)):
            array = basis[index][0]*length + basis[index][1]*length + \
                    padding_basis_left*(half_length - length) + padding_basis_right*(half_length - length)
            offset_array = [-1]*length + [1]*length + \
                    [-1]*(half_length - length) + [1]*(half_length - length)
            array = np.array(array)*sensitivity
            neighbor = array+np.array(offset_array)*sensitivity
            yield array, neighbor


def gen_default_neighboring_pair(query_length, epsilon, sensitivity=1):
    assert query_length % 2 == 0
    assert sensitivity > 0

    half_length = int(query_length/2)
    length = int(epsilon*10+2)
    array = [1]*length + [0]*length + \
            [1]*(half_length - length) + [0]*(half_length - length)
    offset_array = [-1]*length + [1]*length + \
            [-1]*(half_length - length) + [1]*(half_length - length)
    array = np.array(array)*sensitivity
    neighbor = array+np.array(offset_array)*sensitivity
    return array, neighbor


def generate_default_configuration():
    epsilon = 1
    claimed_epsilon = 0.1
    gamma = 0.01

    dataset_settings = {
        'database_1': [2,  0,  1, -1],
        'database_0': [1, 1, 0, 0],
        'sensitivity': 1,
        'epsilon': epsilon,
        'claimed_epsilon': claimed_epsilon,
        'delta': 0.000000001
    }

    kwargs = {
        'dataset_settings': dataset_settings, 'random_seed': int(time.time()),
        'gamma': gamma,
        'training_set_size': 10**7, 'validation_set_size': 10**6,
        'training_batch_size': 10**5, 'testing_batch_size': 10**4,
        'buffer_size': 10**5
    }
    return kwargs

