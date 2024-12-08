import logging
import multiprocessing
import secrets
import time
from functools import partial

import numpy as np
from numpy.random import MT19937, RandomState

from analysis.commons import prune_pos_samples
from classifier.kNN import train_model, stack_samples, compute_testing_risk
from utils.constants import WORKERS, BATCH_SAMPLES
from utils.commons import compute_tight_bound, accuracy_to_delta


class ExpSampleGenerator:
    def __init__(self, kwargs):
        # dataset sanity check
        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]
        self.outcomes_size = kwargs["dataset_settings"]["outcomes_size"]

        assert isinstance(self.X0, np.ndarray), "the database_0 should be in the type of np.ndarray"
        assert isinstance(self.X1, np.ndarray), "the database_1 should be in the type of np.ndarray"
        assert self.X0.ndim == 1, "the database_0 should be in one dimension"
        assert self.X1.ndim == 1, "the database_1 should be in one dimension"
        assert self.X0.dtype <= np.int64, "the database_0's item should be normalized to integer less than the size " \
                                          "of dataset "
        assert self.X1.dtype <= np.int64, "the database_1's item should be normalized to integer less than the size " \
                                          "of dataset "
        assert self.X0.max() <= self.outcomes_size, "the database_0's item should be normalized to integer less than " \
                                                    "the size of dataset "
        assert self.X1.max() <= self.outcomes_size, "the database_1's item should be normalized to integer less than " \
                                                    "the size of dataset "

        # input parameters
        self.sensitivity = kwargs["dataset_settings"]["sensitivity"]
        self.epsilon = kwargs["dataset_settings"]["epsilon"]
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]
        self.dimensionality = 1

        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        self.alternative_sample_noise = 100000000

        # set randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

        # initialize the exponential mechanism applying on mode
        self.quality_table_X0 = self.get_quality_table(self.outcomes_size, self.X0)
        self.quality_table_X1 = self.get_quality_table(self.outcomes_size, self.X1)
        self.probability_table_X0 = self.get_probability_table(self.outcomes_size, self.X0, self.epsilon)
        self.probability_table_X1 = self.get_probability_table(self.outcomes_size, self.X1, self.epsilon)

    @staticmethod
    def get_quality_table(n, dataset):
        quality_table = {}
        for o in np.arange(n):
            quality_table[o] = 0
        for item in dataset:
            quality_table[item] += 1
        return quality_table

    @staticmethod
    def get_probability_table(n, dataset, epsilon):
        # step 0: compute quality table
        quality_table = ExpSampleGenerator.get_quality_table(n, dataset)
        # step 1: compute denominator
        a = 0
        for o in np.arange(n):
            a += np.exp(epsilon / 2 * quality_table[o])
        # step 2: compute probability table
        probability_table = np.zeros(n)
        for o in np.arange(n):
            probability_table[o] = np.exp(epsilon / 2 * quality_table[o]) / a

        return probability_table

    def theoretical_delta(self):
        delta = 0
        for o in np.arange(self.outcomes_size):
            delta += max(0, self.probability_table_X1[o]-np.exp(self.claimed_epsilon)*self.probability_table_X0[o])

        return delta

    def gen_samples(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = np.ones(num_samples)
            X = self.rng.choice(self.outcomes_size, num_samples, p=self.probability_table_X1)

            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            # convert it from boolean to integer
            p = p * 1
            X = (p * X + (1 - p) * self.alternative_sample_noise).reshape((-1, 1))
            return {'X': X.astype(np.float64), 'y': y}
        else:
            y = np.zeros(num_samples)
            X = self.rng.choice(self.outcomes_size, num_samples, p=self.probability_table_X0)
            X = X.reshape((-1, 1))
            return {'X': X.astype(np.float64), 'y': y}


class ExpEstimator:
    def __init__(self, kwargs):
        self.sample_generator = ExpSampleGenerator(kwargs)
        self.training_set_size = kwargs["training_set_size"]
        self.validation_set_size = kwargs["validation_set_size"]
        self.gamma = kwargs["gamma"]
        self.dataset_settings = kwargs["dataset_settings"]

        self.output_ = None
        self.model = None

    def parallel_build(self, classifier="kNN", workers=WORKERS, file_name="nn_files", classifier_args=None):
        logging.info('Generate samples')
        training_pos_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
                                                                 generate_positive_sample=True)
        training_neg_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
                                                                 generate_positive_sample=False)

        testing_pos_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
                                                                generate_positive_sample=True)
        testing_neg_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
                                                                generate_positive_sample=False)

        # Pruned null value
        training_pos_samples['X'] = prune_pos_samples(samples=training_pos_samples['X'],
                                                      dim=self.sample_generator.dimensionality)
        training_pos_samples['y'] = training_pos_samples['y'][:training_pos_samples['X'].shape[0]]
        testing_pos_samples['X'] = prune_pos_samples(samples=testing_pos_samples['X'],
                                                     dim=self.sample_generator.dimensionality)
        testing_pos_samples['y'] = testing_pos_samples['y'][:testing_pos_samples['X'].shape[0]]
        n_null_testing = int(self.validation_set_size / 2) - testing_pos_samples['X'].shape[0]

        tic = time.perf_counter()
        self.model = train_model(positive_samples=training_pos_samples, negative_samples=training_neg_samples,
                                 classifier=classifier, file_name=file_name, n_workers=workers,
                                 classifier_args=classifier_args, permute=True)
        toc = time.perf_counter()
        logging.info(f"Trained {classifier} classifier in {toc - tic:0.4f} seconds")

        logging.info(f'Test {classifier} classifier')
        tic = time.perf_counter()
        #   chunk the testing samples for different workers
        tested_set_size = len(testing_pos_samples['X']) + len(testing_neg_samples['X'])

        n_batch = workers
        batch_samples = int(np.ceil(tested_set_size / n_batch))
        samples_x = np.vsplit(np.vstack((testing_pos_samples['X'], testing_neg_samples['X'])),
                              indices_or_sections=range(batch_samples, tested_set_size, batch_samples))
        samples_y = np.vsplit(np.vstack((testing_pos_samples['y'].reshape((-1, 1)),
                                         testing_neg_samples['y'].reshape((-1, 1)))),
                              indices_or_sections=range(batch_samples, tested_set_size, batch_samples))

        samples = [{'X': samples_x[i], 'y': samples_y[i].ravel()} for i in range(int(n_batch))]

        toc = time.perf_counter()
        logging.info(f"Start Parallel Computation ... {toc - tic:0.4f} seconds")

        pool = multiprocessing.Pool(processes=workers)
        partial_empirical_error_rate = partial(compute_testing_risk, model=self.model)

        results_list = pool.map(partial_empirical_error_rate, samples)
        shaped_accuracy = np.mean(np.array(results_list)).item()
        accuracy = (shaped_accuracy * (
                    self.validation_set_size - n_null_testing) + n_null_testing) / self.validation_set_size

        toc = time.perf_counter()
        logging.critical(f"Parallel Compute the empirical error rate requires {toc - tic:0.4f} seconds")

        logging.info('Compute estimated delta')
        mu = compute_tight_bound(
            gamma=self.gamma, n1=self.training_set_size,
            n2=self.validation_set_size, d=len(self.dataset_settings['database_0']),
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

    def build(self, classifier="kNN"):
        logging.info('Generate samples')
        training_pos_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
                                                                 generate_positive_sample=True)
        training_neg_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
                                                                 generate_positive_sample=False)

        testing_pos_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
                                                                generate_positive_sample=True)
        testing_neg_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
                                                                generate_positive_sample=False)

        logging.info(f"Train {classifier} classifier")
        self.model = train_model(positive_samples=training_pos_samples, negative_samples=training_neg_samples,
                                 classifier=classifier, permute=True)

        logging.info(f"Test {classifier} classifier")
        tic = time.perf_counter()
        samples = stack_samples(positive_samples=testing_pos_samples, negative_samples=testing_neg_samples)
        toc = time.perf_counter()
        logging.info(f"Start Computation ... {toc - tic:0.4f} seconds")

        accuracy = compute_testing_risk(samples=samples, model=self.model)
        toc = time.perf_counter()
        logging.critical(f"Compute the empirical error rate requires {toc - tic:0.4f} seconds")

        logging.info('Compute estimated delta')
        mu = compute_tight_bound(
            gamma=self.gamma, n1=self.training_set_size,
            n2=self.validation_set_size, d=len(self.dataset_settings['database_0']),
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
