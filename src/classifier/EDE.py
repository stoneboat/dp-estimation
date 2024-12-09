import numpy as np


class EmpiricalDistributionEstimator:
    def __init__(self):
        self.histogram = [{}, {}]
        self.cnt = 0
        self.outcomes = []
        self.distribution = None

    def build(self, samples):
        self.histogram = [{}, {}]
        self.cnt = 0
        self.update(samples)

    def update(self, samples):
        y = samples['y']
        X = samples['X']
        assert y.size == X.size
        self.cnt += X.size

        for i in range(y.size):
            try:
                self.histogram[y[i]][X[i]] = self.histogram[y[i]][X[i]] + 1
            except:
                self.histogram[y[i]][X[i]] = 1

        # reset the production
        self.outcomes = []
        self.distribution = None

    def combine_distribution(self, estimator):
        assert isinstance(estimator, EmpiricalDistributionEstimator)
        histogram = estimator.histogram
        self.cnt = self.cnt + estimator.cnt
        for cls_index in range(2):
            if not histogram[cls_index]:
                continue
            for key, value in histogram[cls_index].items():
                try:
                    self.histogram[cls_index][key] = self.histogram[cls_index][key] + value
                except:
                    self.histogram[cls_index][key] = value

        # reset the production
        self.outcomes = []
        self.distribution = None

    def compute_outcomes_list(self):
        outcomes = set(self.histogram[0].keys())
        outcomes.update(set(self.histogram[1].keys()))
        self.outcomes = sorted(outcomes)
        return self.outcomes

    def compute_empirical_distribution(self):
        if self.outcomes:
            outcomes = self.outcomes
        else:
            outcomes = self.compute_outcomes_list()

        distribution = np.zeros((2, len(outcomes)))
        for cls_index in range(2):
            for i in range(len(outcomes)):
                try:
                    distribution[cls_index][i] = self.histogram[cls_index][outcomes[i]]/self.cnt
                except:
                    distribution[cls_index][i] = 0

        self.distribution = distribution
        return self.distribution

    def compute_accuracy(self):
        if self.distribution is None:
            distribution = self.compute_empirical_distribution()
        else:
            distribution = self.distribution

        return np.sum(distribution.max(axis=0))
