import numpy as np
from scipy.special import erf
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def compute_gaussian_mechanism_privacy_over_query(epsilon, sigma, query1, query2):
    """
    Compute the privacy of the additive Gaussian mechanism: takes as input the privacy parameter epsilon,
    the standard deviation sigma, where the Gaussian noise distribution is N(0, I * sigma^2),
    , and the two query functions over a adjacent datasets.
    """

    assert query1.shape[0] == query2.shape[0], "The dimension of the two query functions must be the same"
    assert query1.ndim == 1 and query2.ndim == 1, "The query functions must be vectors"
    assert epsilon >= 0, "The privacy parameter epsilon must be non-negative"
    assert sigma > 0, "The standard deviation sigma must be positive"


    t = np.linalg.norm(query1 - query2, ord=2)/sigma

    term1 = (1 - np.exp(epsilon)) / 2
    sqrt8 = np.sqrt(8)
    sqrt2 = np.sqrt(2)
    term2 = 0.5 * (
        erf((-epsilon / (sqrt2*t)) + (t / sqrt8)) -
        np.exp(epsilon) * erf((-epsilon / (sqrt2*t)) - (t / sqrt8))
    )
    return max(0, term1 + term2)

# Implemented from the result in https://proceedings.mlr.press/v54/li17a.html with title "Minimax Gaussian Classification & Clustering"
def standard_gaussian_BOC(x, mu0, mu1, pi0=0.5, pi1=0.5):
    """
    Optimal Bayes classifier for distinguishing two classes
    with Gaussian distributions N(mu0, sigma^2 I) and N(mu1, sigma^2 I), 
    prior probabilities pi0 and pi1.
    
    Parameters:
        x:   data point (1d numpy array)
        mu0: mean vector of class 0 (1d numpy array)
        mu1: mean vector of class 1 (1d numpy array)
        pi0: prior probability of class 0
        pi1: prior probability of class 1
        
    Returns:
        label: 0 or 1 (Bayes-optimal prediction)
    """
    # Compute the linear score
    score = (mu1 - mu0).T@x
    
    # Compute the bias term (b^*)
    bias = 0.5 * (mu0.T@mu0 - mu1.T@mu1) + np.log(pi1 / pi0)
    
    # Decision rule
    return int(score + bias > 0)

def generate_sparse_linear_classifier_for_standard_gaussian(pos_samples, neg_samples, reg_strength=None, pi0=0.5, pi1=0.5):
    """
    Generate a sparse linear classifier for distinguishing two classes
    with Gaussian distributions N(mu0, sigma^2 I) and N(mu1, sigma^2 I), 
    prior probabilities pi0 and pi1. The classifier is trained on the samples from the two classes.
    
    the sparse linear classifier is of the form w^T x + b > 0 
    where w is estimated by Lasso regression: min ||w - (mu1 - mu0)||_2^2 + reg_strength||w||_1^2,
    and b is estimated by b = -0.5*w^T (mu0 + mu1) + log(pi1 / pi0)

    Assumptions: mu0 and mu1 are unknown, but they differ in at least one dimension.
                 The prior probabilities pi0 and pi1 are known.
    
    Setting of the regularization strength is as the level of sqrt(sparsity * log(dimensions) / sample size), 
    which is accoding to the excess risk bound in "A Direct Estimation Approach to Sparse Linear Discriminant Analysis"
    from https://arxiv.org/abs/1107.3442
        
    Returns:
        sparse_linear_classifier: a function that takes a data point and returns 0 or 1
    """
    mu_hat_1 = np.mean(pos_samples, axis=0)
    mu_hat_0 = np.mean(neg_samples, axis=0)

    dim = len(mu_hat_0)
    n = len(pos_samples) + len(neg_samples)
    C = 2

    if reg_strength is None:
        reg_strength = C * np.sqrt(np.log(dim)/n)
    
    lasso = Lasso(alpha=reg_strength)
    lasso.fit(X=np.identity(len(mu_hat_0)), y=mu_hat_1 - mu_hat_0)
    
    w = lasso.coef_.reshape((-1, 1))
    b = -0.5*w.T@((mu_hat_0).reshape((-1, 1)) + mu_hat_1.reshape((-1, 1))) + np.log(pi1 / pi0)
    w = w.flatten()
    b = b.item()
    
    def sparse_linear_classifier(x):
        return 1 if np.dot(w, x) + b > 0 else 0
    
    return sparse_linear_classifier


def tune_b_in_sparse_linear_classifier(w, X_val, y_val, b_grid=None):
    if b_grid is None:
        # Search over a grid centered around zero
        b_grid = np.linspace(-5, 5, 200)

    errors = []
    for b in b_grid:
        preds = (X_val @ w + b > 0).astype(int)
        error = np.mean(preds != y_val)
        errors.append(error)
    best_b = b_grid[np.argmin(errors)]
    return best_b

def generate_sparse_linear_classifier_for_standard_gaussian_robust(samples, reg_strength=None, test_size=0.7):
    """
    Generate a sparse linear classifier for distinguishing two classes
    with Gaussian distributions N(mu0, sigma^2 I) and N(mu1, sigma^2 I), 
    prior probabilities pi0 and pi1. The classifier is trained on the samples from the two classes.
    
    the sparse linear classifier is of the form w^T x + b > 0 
    where w is estimated by Lasso regression: min ||w - (mu1 - mu0)||_2^2 + reg_strength||w||_1^2.
    This method does not assume priors of pi0 and pi1 or perfect estimates of mu0 and mu1.
    and b is esimtated via cross-validation.

    Assumptions: mu0 and mu1 are unknown, but they differ in at least one dimension.
                 Sample size are imbalanced.
        
    Returns:
        sparse_linear_classifier: a function that takes a data point and returns 0 or 1
    """
    X_train, X_val, y_train, y_val = train_test_split(samples['X'], samples['y'], test_size=test_size)

    mu_hat_1 = np.mean(X_train[y_train == 1], axis=0)
    mu_hat_0 = np.mean(X_train[y_train == 0], axis=0)

    dim = len(mu_hat_0)
    n = len(samples['y'])
    C = 2

    if reg_strength is None:
        reg_strength = C * np.sqrt(np.log(dim)/n)
    
    lasso = Lasso(alpha=reg_strength)
    lasso.fit(X=np.identity(len(mu_hat_0)), y=mu_hat_1 - mu_hat_0)
    w = lasso.coef_
    b = tune_b_in_sparse_linear_classifier(w, X_val, y_val)
    
    def sparse_linear_classifier(x):
        return 1 if np.dot(w, x) + b > 0 else 0
    
    return sparse_linear_classifier, w, b