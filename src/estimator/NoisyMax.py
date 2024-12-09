import logging
import multiprocessing
import secrets
import time
from enum import Enum
from functools import partial

import numpy as np
from numpy.random import MT19937, RandomState

from classifier.kNN import train_model, stack_samples, compute_testing_risk
from utils.constants import WORKERS, BATCH_SAMPLES
from utils.commons import compute_tight_bound, accuracy_to_delta


class NoisyMaxVariant(Enum):
    NM1 = 1
    NM2 = 2
    NM3 = 3
    NM4 = 4


class NoisyMaxSampleGenerator:
    def __init__(self, kwargs):
        # dataset sanity check
        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]
        self.outcomes_size = kwargs["dataset_settings"]["outcomes_size"]

        assert isinstance(self.X0, np.ndarray), "the database_0 should be in the type of np.ndarray"
        assert isinstance(self.X1, np.ndarray), "the database_1 should be in the type of np.ndarray"
        assert self.X0.ndim == 1, "the database_0 should be in one dimension"
        assert self.X1.ndim == 1, "the database_1 should be in one dimension"
        assert self.X1.size == self.X0.size, "the database should have the same length"

        # input parameters
        self.sensitivity = kwargs["dataset_settings"]["sensitivity"]
        self.epsilon = kwargs["dataset_settings"]["epsilon"]
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]
        self.dimensionality = 1
        self.nm_variant = kwargs["nm_variant"]

        self.noisy_scale = self.sensitivity / self.epsilon
        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        self.alternative_sample_noise = 100000000

        # set randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def gen_samples(self, num_samples, generate_positive_sample):
        if self.nm_variant == NoisyMaxVariant.NM1:
            return self.gen_samples_NM1(num_samples, generate_positive_sample)
        if self.nm_variant == NoisyMaxVariant.NM2:
            return self.gen_samples_NM2(num_samples, generate_positive_sample)
        if self.nm_variant == NoisyMaxVariant.NM3:
            return self.gen_samples_NM3(num_samples, generate_positive_sample)
        if self.nm_variant == NoisyMaxVariant.NM4:
            return self.gen_samples_NM4(num_samples, generate_positive_sample)

    def gen_samples_NM1(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = np.ones(num_samples)
            X = self.X1
            noise = self.rng.laplace(loc=0, scale=self.noisy_scale, size=(num_samples, X.size))
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            # convert it from boolean to integer
            p = p * 1
            return {'X': p * np.argmax(X + noise, axis=1) + (1 - p) * self.alternative_sample_noise, 'y': y}
        else:
            y = np.zeros(num_samples)
            X = self.X0
            noise = self.rng.laplace(loc=0, scale=self.noisy_scale, size=(num_samples, X.size))
            return {'X': np.argmax(X + noise, axis=1), 'y': y}

    def gen_samples_NM2(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = np.ones(num_samples)
            X = self.X1
            noise = self.rng.laplace(loc=0, scale=self.noisy_scale, size=(num_samples, X.size))
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            # convert it from boolean to integer
            p = p * 1
            return {'X': p * np.max(X + noise, axis=1) + (1 - p) * self.alternative_sample_noise, 'y': y}
        else:
            y = np.zeros(num_samples)
            X = self.X0
            noise = self.rng.laplace(loc=0, scale=self.noisy_scale, size=(num_samples, X.size))
            return {'X': np.max(X + noise, axis=1), 'y': y}

    def gen_samples_NM3(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = np.ones(num_samples)
            X = self.X1
            noise = self.rng.exponential(scale=self.noisy_scale, size=(num_samples, X.size))
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            # convert it from boolean to integer
            p = p * 1
            return {'X': p * np.argmax(X + noise, axis=1) + (1 - p) * self.alternative_sample_noise, 'y': y}
        else:
            y = np.zeros(num_samples)
            X = self.X0
            noise = self.rng.exponential(scale=self.noisy_scale, size=(num_samples, X.size))
            return {'X': np.argmax(X + noise, axis=1), 'y': y}

    def gen_samples_NM4(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = np.ones(num_samples)
            X = self.X1
            noise = self.rng.exponential(scale=self.noisy_scale, size=(num_samples, X.size))
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            # convert it from boolean to integer
            p = p * 1
            return {'X': p * np.max(X + noise, axis=1) + (1 - p) * self.alternative_sample_noise, 'y': y}
        else:
            y = np.zeros(num_samples)
            X = self.X0
            noise = self.rng.exponential(scale=self.noisy_scale, size=(num_samples, X.size))
            return {'X': np.max(X + noise, axis=1), 'y': y}


class NoisyMaxEstimator:
    def __init__(self, kwargs):
        self.sample_generator = NoisyMaxSampleGenerator(kwargs)
        self.training_set_size = kwargs["training_set_size"]
        self.validation_set_size = kwargs["validation_set_size"]
        self.gamma = kwargs["gamma"]
        self.dataset_settings = kwargs["dataset_settings"]

        self.output_ = None
        self.model = None

    def parallel_build(self, workers=WORKERS):
        logging.info('Generate samples')
        training_pos_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
                                                                 generate_positive_sample=True)
        training_neg_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
                                                                 generate_positive_sample=False)

        testing_pos_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
                                                                generate_positive_sample=True)
        testing_neg_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
                                                                generate_positive_sample=False)

        logging.info('Train kNN classifier')
        self.model = train_model(positive_samples=training_pos_samples, negative_samples=training_neg_samples,
                                 permute=True)

        logging.info('Test kNN classifier')
        tic = time.perf_counter()
        #   chunk the testing samples for different workers
        tested_set_size = len(testing_pos_samples['X']) + len(testing_neg_samples['X'])

        n_batch = workers
        batch_samples = int(np.ceil(tested_set_size / n_batch))
        samples_x = np.hsplit(np.hstack((testing_pos_samples['X'], testing_neg_samples['X'])),
                              indices_or_sections=range(batch_samples, tested_set_size, batch_samples))
        samples_y = np.hsplit(np.hstack((testing_pos_samples['y'], testing_neg_samples['y'])),
                              indices_or_sections=range(batch_samples, tested_set_size, batch_samples))

        samples = [{'X': samples_x[i], 'y': samples_y[i].ravel()} for i in range(int(n_batch))]

        toc = time.perf_counter()
        logging.info(f"Start Parallel Computation ... {toc - tic:0.4f} seconds")

        pool = multiprocessing.Pool(processes=workers)
        partial_empirical_error_rate = partial(compute_testing_risk, model=self.model)

        results_list = pool.map(partial_empirical_error_rate, samples)
        accuracy = np.mean(np.array(results_list))
        toc = time.perf_counter()
        logging.critical(f"Parallel Compute the empirical error rate requires {toc - tic:0.4f} seconds")

        logging.info('Compute estimated delta')

        mu = compute_tight_bound(
            gamma=self.gamma, n1=self.training_set_size,
            n2=self.validation_set_size, d=1,
            epsilon=self.dataset_settings['claimed_epsilon']
        )

        estimated_delta = accuracy_to_delta(accuracy, self.dataset_settings['claimed_epsilon'])

        estimated_range = (max(0, estimated_delta - mu), max(estimated_delta + mu, 0))
        self.output_ = {
            'estimated_delta': estimated_delta, 'accuracy': accuracy,
            'estimated_range': estimated_range, 'gamma': self.gamma,
            'training_set_size': self.training_set_size, 'validation_set_size': self.validation_set_size
            # 'error ratio': err1/err2
        }

        return self.output_

    def build(self):
        logging.info('Generate samples')
        training_pos_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
                                                                 generate_positive_sample=True)
        training_neg_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
                                                                 generate_positive_sample=False)

        testing_pos_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
                                                                generate_positive_sample=True)
        testing_neg_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
                                                                generate_positive_sample=False)

        logging.info('Train kNN classifier')
        self.model = train_model(positive_samples=training_pos_samples, negative_samples=training_neg_samples,
                                 permute=True)

        logging.info('Test kNN classifier')
        tic = time.perf_counter()
        samples = stack_samples(positive_samples=testing_pos_samples, negative_samples=testing_neg_samples)
        toc = time.perf_counter()
        logging.info(f"Start Parallel Computation ... {toc - tic:0.4f} seconds")

        accuracy = compute_testing_risk(samples=samples, model=self.model)
        toc = time.perf_counter()
        logging.critical(f"Parallel Compute the empirical error rate requires {toc - tic:0.4f} seconds")

        logging.info('Compute estimated delta')
        mu = compute_tight_bound(
            gamma=self.gamma, n1=self.training_set_size,
            n2=self.validation_set_size, d=1,
            epsilon=self.dataset_settings['claimed_epsilon']
        )

        estimated_delta = accuracy_to_delta(accuracy, self.dataset_settings['claimed_epsilon'])
        estimated_range = (max(0, estimated_delta - mu), max(estimated_delta + mu, 0))
        self.output_ = {
            'estimated_delta': estimated_delta, 'accuracy': accuracy,
            'estimated_range': estimated_range, 'gamma': self.gamma,
            'training_set_size': self.training_set_size, 'validation_set_size': self.validation_set_size
        }

        return self.output_


def gen_demo_test_neighbors(nm_variant):
    if nm_variant == NoisyMaxVariant.NM1:
        return np.array([1, 1, 1, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0])
    if nm_variant == NoisyMaxVariant.NM2:
        return np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0])
    if nm_variant == NoisyMaxVariant.NM3:
        return np.array([1, 1, 1, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0])
    if nm_variant == NoisyMaxVariant.NM4:
        return np.array([1, 1]), np.array([0, 0])