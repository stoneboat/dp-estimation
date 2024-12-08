# This is an implementation of the empirical bootstrap as described in Section 6 of
# Jeremy Orloff and Jonathan Bloom "Bootstrap Confidence Intervals",
# Lecture notes for MIT Class 18.05
# Accesed from https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and
# -statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf

from collections import namedtuple

import numpy as np
from sklearn.utils import resample


class EmpiricalBootstrap:
    _RTV = namedtuple('bootstrap_result',
                      ['lb_one_sided', 'lb_two_sided', 'ub_one_sided', 'ub_two_sided'])

    def __init__(self,
                 sample_generator: 'SampleGenerator'):
        self.sample_generator = sample_generator
        self.generated_samples = []

    def delta_means(self, n_samples):
        bootstrap_samples = self.get_bootstrap_means(n_samples)
        sample_mean = self.sample_generator.sample_mean
        return bootstrap_samples - sample_mean

    def get_bootstrap_means(self, n_samples: int) -> np.array:
        n_needed_to_generate = n_samples - len(self.generated_samples)
        for _ in range(n_needed_to_generate):
            self.generated_samples.append(self.sample_generator.new_bootstrap_mean())

        return np.array(self.generated_samples[0:n_samples])

    def bootstrap_confidence_bounds(self, confidence_interval_prob, n_samples) -> _RTV:
        delta_means = self.delta_means(n_samples)
        sample_mean = self.sample_generator.sample_mean

        one_sided_error_percentage = (1 - confidence_interval_prob) * 100
        two_sided_error_percentage = (1 - confidence_interval_prob) / 2 * 100
        # eg if confidence_inrvela_prob = 0.9
        # error percentage is 0.1
        # P(mu >= x - d_0.9) >= 0.9
        lb_one_sided = sample_mean - np.percentile(
            delta_means, q=100 - one_sided_error_percentage, interpolation="higher"
        )
        # P(mu <= x - d_0.1) >= 0.9
        # error percentage is 0.1
        ub_one_sided = sample_mean - np.percentile(
            delta_means, q=one_sided_error_percentage, interpolation="lower"
        )
        lb_two_sided = sample_mean - np.percentile(
            delta_means, q=100 - two_sided_error_percentage, interpolation="higher"
        )
        ub_two_sided = sample_mean - np.percentile(
            delta_means, q=two_sided_error_percentage, interpolation="lower"
        )
        return self._RTV(lb_one_sided, lb_two_sided, ub_one_sided, ub_two_sided)


class SampleGenerator:
    def __init__(self, data, seed=0):
        self.data = data
        self.sample_mean = np.mean(self.data)
        self._random_state = np.random.RandomState(seed)

    def new_bootstrap_mean(self):
        data = self.new_bootstrap_sample()
        return np.mean(data)

    def new_bootstrap_sample(self):
        data = resample(self.data, random_state=self._random_state)
        return data
