import logging
import multiprocessing
import time
from functools import partial

from analysis.commons import prune_pos_samples
from mech.LaplaceMechanism import LaplaceSampleGenerator
from classifier.kNN import train_model, compute_testing_risk, compute_shaping_risk, compute_correct_num, stack_samples
from utils.constants import BATCH_SAMPLES, WORKERS
from utils.commons import compute_tight_bound, accuracy_to_delta, compute_improved_tight_bound_tuple

import numpy as np


class LaplaceEstimator:
    def __init__(self, kwargs):
        self.sample_generator = LaplaceSampleGenerator(kwargs)
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
                                 classifier=classifier, file_name=file_name, n_workers=workers, classifier_args=classifier_args)
        toc = time.perf_counter()
        logging.info(f"Trained {classifier} classifier in {toc - tic:0.4f} seconds")

        logging.info(f'Test {classifier} classifier')
        tic = time.perf_counter()
        #   chunk the testing samples for different workers
        tested_set_size = len(testing_pos_samples['X']) + len(testing_neg_samples['X'])

        n_batch = workers
        batch_samples = int(np.ceil(tested_set_size/n_batch))
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
        accuracy = (shaped_accuracy*(self.validation_set_size-n_null_testing) + n_null_testing)/self.validation_set_size

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
                                 classifier=classifier)

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


