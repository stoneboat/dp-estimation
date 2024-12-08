from math import ceil, sqrt
from math import exp, log

import numpy as np

def accuracy_to_delta(accuracy, epsilon):
    return max(0, 1-2*exp(epsilon)*(1-accuracy))


def compute_tight_bound(gamma, n1, n2, d, epsilon):
    """The tight bound states that with probability at least 1 - gamma, given n samples, the mechanism (
    having a density) with dimensionality d, the estimator could output a tight bound within additive error
    u (the function's return value)"""
    assert ((gamma <= 1) & (gamma > 0))
    assert ((n1 > 0) & (n2 > 0) & (d >= 1))

    cd = pow(1 + 2 / sqrt(2 - sqrt(3)), d) - 1
    err1 = 12 * cd * sqrt(log(4 / gamma) / (2 * n1))
    err2 = sqrt(1 / (2 * n2) * log(4 / gamma))

    return 2 * exp(epsilon) * (err1 + err2)


def compute_improved_tight_bound_tuple(gamma, n1, n2, d):
    """The tight bound states that with probability at least 1 - gamma, given n samples, the mechanism (
    having a density) with dimensionality d, the estimator could output a tight bound within additive error
    u (the function's return value)"""
    assert ((gamma <= 1) & (gamma > 0))
    assert ((n1 > 0) & (n2 > 0) & (d >= 1))

    cd = pow(1 + 2 / sqrt(2 - sqrt(3)), d) - 1
    err1 = 12 * cd * sqrt(log(4 / gamma) / (2 * n1))
    err2 = sqrt(1 / (2 * n2) * log(4 / gamma))

    return err1, err2


def convert_bytes_to_mb(x):
    assert np.isreal(x)
    return x / 1048576


def convert_mb_to_bytes(x):
    assert np.isreal(x)
    return x * 1048576


def convert_bytes_to_gb(x):
    assert np.isreal(x)
    return x / 1073741824


def convert_gb_to_bytes(x):
    assert np.isreal(x)
    return x * 1073741824