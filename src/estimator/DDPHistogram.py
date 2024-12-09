import logging
import multiprocessing
import numpy as np
import secrets
import time
from functools import partial

from numpy.random import MT19937, RandomState
from scipy.stats import binom

from analysis.commons import prune_pos_samples
from classifier.kNN import train_model, stack_samples, compute_testing_risk, stack_parallel_samples
from utils.constants import WORKERS, BATCH_SAMPLES
from utils.commons import compute_tight_bound, accuracy_to_delta


class DDPHistogramSampleGenerator:
    def __init__(self, kwargs):
        self.bins_size = kwargs["dataset_settings"]["bins_size"]
        self.voter_number = kwargs["dataset_settings"]["voter_number"]
        assert isinstance(self.bins_size, int) and self.bins_size > 0, "bin size shall be an positive integer"
        assert isinstance(self.voter_number, int) and self.voter_number > 0, "number of voter shall be an positive integer"

        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]
        assert self.X0 < self.bins_size, f"voter 0 shall choose a candiate from 0 to {self.bins_size}"
        assert self.X1 < self.bins_size, f"voter 1 shall choose a candiate from 0 to {self.bins_size}"

        # input parameters
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]
        if self.bins_size == 2:
            self.dimensionality = 1
        else:
            self.dimensionality = self.bins_size
        # the histogram is sampled from uniform distribution
        self.p = 0.5

        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        self.alternative_sample_noise = 100000000

        # set randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def gen_samples_for_twobins(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = np.ones(num_samples)
            X = self.rng.binomial(self.voter_number - 1, self.p, num_samples) + self.X1
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            X = p * X + (1 - p) * self.alternative_sample_noise
            return {'X': X.astype(np.float64) / self.voter_number, 'y': y}
        else:
            y = np.zeros(num_samples)
            X = self.rng.binomial(self.voter_number-1, self.p, num_samples) + self.X0
            return {'X': X.astype(np.float64) / self.voter_number, 'y': y}

    def gen_samples(self, num_samples, generate_positive_sample):
        if self.bins_size == 2:
            return self.gen_samples_for_twobins(num_samples, generate_positive_sample)
        if generate_positive_sample:
            y = np.ones(num_samples)
            X = np.zeros((num_samples, self.bins_size))
            votes = self.rng.randint(self.bins_size, size=num_samples * (self.voter_number - 1))
            cnt = 0
            for sample_id in range(num_samples):
                X[sample_id][self.X1] += 1
                for _ in range(self.voter_number - 1):
                    X[sample_id][votes[cnt]] += 1
                    cnt += 1

            p = np.random.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            p = p.reshape((num_samples, 1)) * np.ones((num_samples, self.bins_size))
            X = p*X + (1 - p) * self.alternative_sample_noise
            return {'X': X.astype(np.float64)/self.voter_number, 'y': y}
        else:
            y = np.zeros(num_samples)
            X = np.zeros((num_samples, self.bins_size))
            votes = self.rng.randint(self.bins_size, size=num_samples*(self.voter_number-1))
            cnt = 0
            for sample_id in range(num_samples):
                X[sample_id][self.X0] += 1
                for _ in range(self.voter_number-1):
                    X[sample_id][votes[cnt]] += 1
                    cnt += 1
            return {'X': X.astype(np.float64)/self.voter_number, 'y': y}

    def parallel_gen_samples_class_label_first(self, generate_positive_sample, num_samples):
        """ If you want to use then gen_samples method in multiple processing,
            keep in mind that each process's copy should have fresh randomness, otherwise just accumulate error rather
            than accuracy
        """
        self.reset_randomness()
        return self.gen_samples(num_samples, generate_positive_sample)


class DDPHistogramEstimator:
    def __init__(self, kwargs):
        self.sample_generator = DDPHistogramSampleGenerator(kwargs)
        self.training_set_size = kwargs["training_set_size"]
        self.validation_set_size = kwargs["validation_set_size"]
        self.gamma = kwargs["gamma"]
        self.dataset_settings = kwargs["dataset_settings"]

        self.output_ = None
        self.model = None

    def check_sample_generator(self):
        assert self.sample_generator is not None, "ERR: you need to use a super class or to define the sample " \
                                                  "generator first"

    def parallel_gen_samples(self, generate_positive_sample, num_samples, workers):
        pool = multiprocessing.Pool(processes=workers)

        sample_generating_func = partial(self.sample_generator.parallel_gen_samples_class_label_first,
                                         num_samples=int(num_samples / workers))

        if generate_positive_sample:
            input_list = np.ones(workers).astype(dtype=bool)
        else:
            input_list = np.zeros(workers).astype(dtype=bool)
        return stack_parallel_samples(pool.map(sample_generating_func, input_list))


    def parallel_build(self, classifier="kNN", workers=WORKERS, file_name="nn_files", classifier_args=None):
        tic = time.perf_counter()
        # training_pos_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
        #                                                          generate_positive_sample=True)
        # training_neg_samples = self.sample_generator.gen_samples(int(self.training_set_size / 2),
        #                                                          generate_positive_sample=False)
        #
        # testing_pos_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
        #                                                         generate_positive_sample=True)
        # testing_neg_samples = self.sample_generator.gen_samples(int(self.validation_set_size / 2),
        #                                                         generate_positive_sample=False)

        training_pos_samples = self.parallel_gen_samples(num_samples=int(self.training_set_size / 2), workers=workers,
                                                         generate_positive_sample=True)

        training_neg_samples = self.parallel_gen_samples(num_samples=int(self.training_set_size / 2), workers=workers,
                                                         generate_positive_sample=False)

        testing_pos_samples = self.parallel_gen_samples(num_samples=int(self.validation_set_size / 2), workers=workers,
                                                         generate_positive_sample=True)

        testing_neg_samples = self.parallel_gen_samples(num_samples=int(self.validation_set_size / 2), workers=workers,
                                                         generate_positive_sample=False)
        toc = time.perf_counter()
        logging.info(f'Generated samples in  {toc - tic:0.4f} seconds')

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
                                 classifier_args=classifier_args)
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
            n2=self.validation_set_size, d=self.sample_generator.dimensionality,
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
            n2=self.validation_set_size, d=self.sample_generator.dimensionality,
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


def compute_theoretical_delta(n_voters, claimed_epsilon, n_bins):
    assert n_bins == 2, "can compute only for two bins"
    p = 0.5
    delta = 0
    for votes in range(n_voters+1):
        delta += max(0, binom.pmf(votes-1, n_voters-1, p)-np.exp(claimed_epsilon)*binom.pmf(votes, n_voters-1, p))

    return delta
