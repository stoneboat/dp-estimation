import secrets
import time

import numpy as np
from numpy.random import MT19937, RandomState


class LaplaceSampleGenerator:
    """
    Class parameters
    X0, X1: neighbouring databases
    dimensionality: size of the database
    sensitivity
    epsilon
    claimed_epsilon

    laplace_scale
    probability_of_natural_sample
    probability_of_alternative_sample
    alternative_sample_noise
    """

    def __init__(self, kwargs):
        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]

        if not isinstance(self.X0, np.ndarray):
            self.X0 = np.array(self.X0)
        if not isinstance(self.X1, np.ndarray):
            self.X1 = np.array(self.X1)
        self.X0 = np.reshape(self.X0, (1, self.X0.size))
        self.X1 = np.reshape(self.X1, (1, self.X1.size))

        self.sensitivity = kwargs["dataset_settings"]["sensitivity"]
        self.one = np.ones((1,))
        self.zero = np.zeros((1,))
        self.epsilon = kwargs["dataset_settings"]["epsilon"]
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]

        assert (self.X1.size == self.X0.size)
        self.dimensionality = self.X1.size

        # samples are generated according to the actual epsilon
        # but the experiment calls for natural samples to be drawn with
        # probability reciprocal to the claimed epsilon rather than the
        # actual epsilon
        self.laplace_scale = self.sensitivity / self.epsilon
        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        # we output an alternate sample that has negligible probability
        # (approx exp(-1000)) of being generated by the laplace distribution
        self.alternative_sample_noise = (
                -10000000 * self.laplace_scale * np.ones_like(self.X1)
        )

        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def gen_samples(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            n = num_samples
            X = self.X1
            y = self.one * np.ones(n)
            noise = self.rng.laplace(loc=0, scale=self.laplace_scale, size=(n, X.size))
            p = np.random.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            p = p.reshape((num_samples, 1)) * np.ones((num_samples, X.size))
            return {'X': X + (p * noise + (1 - p) * self.alternative_sample_noise), 'y':  y}
        else:
            n = num_samples
            X = self.X0
            y = self.zero * np.ones(n)
            noise = self.rng.laplace(loc=0, scale=self.laplace_scale, size=(n, X.size))
            return {'X': X + noise, 'y': y}