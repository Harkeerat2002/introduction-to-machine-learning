from copy import deepcopy
import numpy as np
import time

import sklearn
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

"""
Generates the data by sampling it from a centered Gaussian distribution with a
certain covariance matrix. It also generates the ground truth function and the
values of the predictor variable y for all the samples.

n = number of samples to be generated
d = dimension of the data
snr = square of the norm of the ground truth function beta_star
noise_sigma = variance of the additive noise added to the predictor variable y
cov = selects the data distribution; can take the values "isotropic" or "misspecified"
"""
def generate_data(n, d, snr, noise_sigma, cov="isotropic", seed=None):
  np.random.seed(seed)
  if cov == "misspecified":
    # Covariance matrix for the misspecified model.
    # For more details see https://arxiv.org/pdf/1903.08560.pdf, Section 5.4.
    latent_size = min(10, d)
    W = get_w_for_latent_space_model(d, latent_size=latent_size)
    Sigma = np.eye(d) + W @ W.T
    theta = np.random.normal(size=(latent_size, 1))
    theta = theta / np.linalg.norm(theta) * np.sqrt(snr)
    inv_ww = np.linalg.inv(np.eye(latent_size) + W.T @ W)
    beta_star = W @ inv_ww @ theta
    noise_sigma = np.sqrt((theta.T @ inv_ww @ theta).reshape(-1)[0])
  elif cov == "isotropic":
    # The covariance matrix is a d-dimensional isotropic Gaussian.
    Sigma = np.eye(d)
    beta_star = np.random.normal(size=(d, 1))
    beta_star = beta_star / np.linalg.norm(beta_star) * np.sqrt(snr)
  else:
    raise RuntimeError("Unrecognized covariance model.")

  X = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
  noise = np.random.normal(scale=noise_sigma, size=(n, 1))
  y = X @ beta_star + noise

  return X, y, beta_star, Sigma

"""
Computes the estimation error ||beta_star - beta_hat||^2 for a linear predictor
beta_hat, when the data is sampled from a Gaussian distribution with 0 mean and
covariance matrix Sigma. The predictor variable y is a linear combination of
the covariates (through the ground truth beta_star) plus additive noise sampled
from N(0, noise_sigma). The closed form of the error in this setting is taken
from https://arxiv.org/pdf/1903.08560.pdf, Section 2.1.
"""

def compute_estimation_error(beta_star, beta_hat, noise_sigma, Sigma):
  beta_star = beta_star.reshape(-1, 1)
  beta_hat = beta_hat.reshape(-1, 1)
  diff = beta_star - beta_hat
  return (diff.T @ Sigma @ diff)[0][0]

def compute_population_risk(beta_star, beta_hat, noise_sigma, Sigma):
  return compute_estimation_error(beta_star, beta_hat, noise_sigma, Sigma) + noise_sigma**2

"""
Computes the empirical test error of a linear predictor beta_hat using a finite
sample test set.
"""
def compute_test_error(beta_hat, X, y):
  pred = X @ np.array(beta_hat).reshape(-1, 1)
  return np.linalg.norm(y - pred)**2 / X.shape[0]

"""
Helper function used to generate data in the misspecified distribution model.
"""
def get_w_for_latent_space_model(d, latent_size):
  W = np.random.normal(size=(d, latent_size))
  svd = np.linalg.svd(W)
  svals = np.zeros_like(W)
  # Set the singular values of W to be equal.
  svals[:latent_size, :latent_size] = np.eye(latent_size) * np.sqrt(d / latent_size)
  W = svd[0] @ svals @ svd[2]
  return W

"""
Helper function that repets an experiment (i.e. calls a function passed as
parameter) a number of times and aggregates the results by averaging the
respective metrics produced by the experiment.
"""
def repeat_experiment(num_runs, func, func_args={}):
  all_outputs = []
  for i in range(num_runs):
    outputs = func(**func_args)
    all_outputs.append(outputs)

  aggregated_outputs = {}
  for key in all_outputs[0].keys():
    aggregated_outputs[key] = np.array(
        [outputs[key] for outputs in all_outputs]).mean(axis=0)

  return aggregated_outputs

"""
Generates additional samples from a given distribution given by the covariance
matrix Sigma and ground truth model given by beta_star and noise_sigma.
"""
def generate_additional_data(num_samples, d, Sigma, beta_star, noise_sigma):
  X = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=num_samples)
  noise = np.random.normal(scale=noise_sigma, size=(num_samples, 1))
  y = X @ beta_star + noise
  return X, y

"""
Computes the error ||beta_star - beta_hat||^2 of the empirical minimizer for
different degrees of overparameterization.

The amount of overparameterization is controled through
the ratio gamma = d / n. In order to vary gamma, one can either fix the data
dimension and vary the number of samples, or the other way round.
If use_ridge is set, then for each level of overparameterization, it determines
the best ridge coefficient using cross-validation and returns the error of the
estimator learned with that coefficient.
"""
def get_error_vs_overparametrization(all_gammas, all_snr, fix_n_vary_d=False,
        n=200, d=1000, cov_type="isotropic", use_ridge=False, noise_sigma=1):
  errors = {}

  n, d = int(n), int(d)
  if not fix_n_vary_d:
    max_n = int(d / np.min(all_gammas)) + 1

  for snr in all_snr:
    errors[snr] = []
    if not fix_n_vary_d:
      X, y, beta_star, Sigma = generate_data(n=max_n, d=d, snr=snr, noise_sigma=noise_sigma, cov=cov_type)
    for gamma in all_gammas:
      if fix_n_vary_d:
        d = int(n * gamma)
        X, y, beta_star, Sigma = generate_data(n=n, d=d, snr=snr, noise_sigma=noise_sigma, cov=cov_type)
      else:
        n = int(d / gamma)
      if use_ridge:
        estimator = RidgeCV(alphas=[0.1, 1, 10], fit_intercept=False, cv=5).fit(X[:n], y[:n])
      else:
        estimator = LinearRegression(fit_intercept=False).fit(X[:n], y[:n])
      errors[snr].append(compute_estimation_error(beta_star, estimator.coef_, noise_sigma, Sigma))
  return errors
