
# dp-estimation

**Eureka: A General Framework for Black-box Differential Privacy Estimators**

This repository provides a proof-of-concept implementation of the black-box differential privacy estimator framework proposed in our paper:

**[Eureka: A General Framework for Black-box Differential Privacy Estimators](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a166/1Ub24GW3Sso)**

## Overview

Differential Privacy (DP) is a rigorous mathematical framework that provides formal privacy guarantees for data analysis. However, deriving privacy parameters analytically can be challenging when dealing with complex or unknown mechanisms. Our work, *Eureka*, introduces a general black-box approach for estimating these parameters. Instead of relying on explicit distributions or proofs, the estimator estimates privacy parameters by interacting with the mechanism as an oracle, observing only its inputs and outputs.

This repository demonstrates two implementations of the black-box estimator framework:
1. A **k-Nearest Neighbor (kNN)** classifier-based estimator.
2. A **Neural Network (NN)** classifier-based estimator.

We use these estimators to plot the **DP privacy spectrum** of several well-known mechanisms and their variants, including:
- Gaussian Mechanism
- Laplacian Mechanism
- Exponential Mechanism
- Noisy Histogram
- Noisy Max
- Sparse Vector Technique (SVT)

Additionally, for the first time, we plot the **DDP privacy spectrum** of a DDP histogram algorithm.

## Key Features

- **Black-box Estimation:**  
  Requires minimal knowledge of the underlying distribution or code structure of the mechanism.  
- **Classifier-based Framework:**  
  Provides flexibility to use different binary classification algorithms. In addition to kNN and NN, other classifiers can be seamlessly integrated.  
- **Broad Applicability:**  
  Supports the evaluation of standard and complex DP mechanisms, helping to identify subtle bugs or test privacy properties.  
- **Comprehensive Demonstrations:**  
  Includes a series of Jupyter notebooks that provide end-to-end demonstrations for privacy estimation on various tested algorithms.


## Getting Started

### Prerequisites
- **Python Version:** Python 3.8+ is recommended.
- **Required Libraries:** Install the following common data science and machine learning libraries:
  - `numpy`, `scipy`, `scikit-learn`, `matplotlib`
  - `torch` (for Neural Network-based classifiers)

### Running the Examples
To explore the functionality of the estimators and learn how to run the code, navigate to the `notebooks` folder and execute the provided Jupyter notebooks. These examples demonstrate the interface of the estimators and their application to various mechanisms.

### Customization

- **Adding New Mechanisms:**  
  To extend the framework to support a new mechanism, implement it in the `src/mech/` directory and create a corresponding estimator in `src/estimator/`.

- **Using Alternative Classifiers:**  
  The framework is modular, allowing you to integrate custom classifiers. Follow the interface defined in `src/classifier/` to add your own classifier.

- **Parameter Tuning:**  
  Experiment with different parameters, including classifier settings, the number of samples, input database configurations, and other hyperparameters. The Jupyter notebooks provide an interactive environment to observe how these adjustments impact estimation quality.

## Citation

If you find this work helpful, please cite our paper: **[Eureka: A General Framework for Black-box Differential Privacy Estimators](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a166/1Ub24GW3Sso)**. 

## Contact 
For questions or comments, please open an issue on GitHub or contact the authors at ywei368@gatech.edu.
