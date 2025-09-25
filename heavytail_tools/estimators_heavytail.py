import numpy as np
import torch
from typing import List, Union


def hill_estimator(data: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the Hill estimator for the tail index of each column in the dataset.

    Args:
        data (np.ndarray): Input data of shape (N, d), all values must be positive.
        k (int): Number of top order statistics to use.

    Returns:
        np.ndarray: Hill estimates for each column (shape: d,).
    """
    data = np.asarray(data)
    if np.any(data <= 0):
        raise ValueError("All data values must be positive.")

    n_samples, n_features = data.shape
    hill_indices: List[float] = []

    for col_idx in range(n_features):
        col_values = data[:, col_idx]
        sorted_col = np.sort(col_values)
        tail_values = sorted_col[-k:]
        x_k_plus1 = sorted_col[-k - 1]
        log_ratios = np.log(tail_values) - np.log(x_k_plus1 + 1e-12)
        hill_value = (1 / k) * np.sum(log_ratios)
        hill_indices.append(1 / hill_value)

    return np.array(hill_indices)


def quantile_estimator(data: np.ndarray,
                       q_s: Union[List[float], np.ndarray] = [0.1, 0.5, 0.8, 0.9, 0.99, 0.999]) -> np.ndarray:
    """
    Estimates empirical quantiles of the data.

    Args:
        data (np.ndarray): Input data array.
        q_s (list or np.ndarray): Quantiles to compute (values between 0 and 1).

    Returns:
        np.ndarray: Estimated quantiles.
    """
    q_s = np.asarray(q_s)
    if np.any((q_s < 0) | (q_s > 1)):
        raise ValueError("Quantiles must be between 0 and 1.")

    return np.quantile(data, q_s, axis=0)


def frechet_cdf(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Computes the CDF of a Frechet distribution for given values and shape parameter alpha.

    Args:
        x (np.ndarray): Values at which to compute the CDF.
        alpha (float): Shape parameter (>0).

    Returns:
        np.ndarray: CDF values.
    """
    x = np.asarray(x)
    cdf_values = np.zeros_like(x, dtype=np.float64)
    positive_mask = x > 0
    cdf_values[positive_mask] = np.exp(-x[positive_mask] ** (-alpha))
    cdf_values[~positive_mask] = 0.0
    return cdf_values


def conditional_tail_prob_frechet(data: np.ndarray, alpha: np.ndarray, i: int, j: int,
                                  t_array: np.ndarray) -> np.ndarray:
    """
    Computes the conditional tail probability P(X_i > t | X_j > t) for Frechet-distributed columns.

    Args:
        data (np.ndarray): Input data array of shape (N, d).
        alpha (np.ndarray): Shape parameters for each column (length d).
        i (int): Index of the first column.
        j (int): Index of the second column.
        t_array (np.ndarray): Thresholds at which to compute the conditional probability.

    Returns:
        np.ndarray: Conditional probabilities for each threshold.
    """
    u_i = frechet_cdf(data[:, i], alpha[i])
    u_j = frechet_cdf(data[:, j], alpha[j])
    t_array = np.asarray(t_array)
    t_uniform = np.zeros_like(t_array, dtype=np.float64)

    positive_mask = t_array > 0
    t_uniform[positive_mask] = np.exp(-(t_array[positive_mask]) ** (-alpha[j]))
    t_uniform[~positive_mask] = 0.0

    mask_i = u_i[:, np.newaxis] > t_uniform[np.newaxis, :]
    mask_j = u_j[:, np.newaxis] > t_uniform[np.newaxis, :]

    n_j = mask_j.sum(axis=0)
    n_i_and_j = np.logical_and(mask_i, mask_j).sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = n_i_and_j / n_j
        result[n_j == 0] = np.nan

    return result


def log_frechet_quantile(alpha: float,
                         q_s: Union[List[float], np.ndarray] = [0.1, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999]) -> np.ndarray:
    """
    Computes the quantiles of the standard Frechet distribution in log-scale.

    Args:
        alpha (float): Shape parameter (>0).
        q_s (list or np.ndarray): Quantiles to compute (0 < q < 1).

    Returns:
        np.ndarray: Log-Frechet quantiles.
    """
    q_s = np.asarray(q_s)
    if np.any((q_s <= 0) | (q_s >= 1)):
        raise ValueError("Quantiles must be strictly between 0 and 1.")
    if alpha <= 0:
        raise ValueError("Alpha must be positive.")

    return - (1 / alpha) * np.log(-np.log(q_s))


def tail_probability_mc(data: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """
    Estimates tail probabilities P(X > x) via Monte Carlo for each threshold in x_values.

    Args:
        data (np.ndarray): Input data array (N, d) or (N,).
        x_values (np.ndarray): Thresholds at which to estimate the tail probability.

    Returns:
        np.ndarray: Tail probabilities for each threshold.
    """
    data = np.asarray(data)
    x_values = np.asarray(x_values)
    exceedances = data[:, None] > x_values[None, :]
    probs = exceedances.mean(axis=0)
    return probs


def tail_prob_frechet_th(t: np.ndarray, alpha: float) -> np.ndarray:
    """
    Computes the theoretical tail probability P(X > t) for a Frechet distribution.

    Args:
        t (np.ndarray): Threshold values.
        alpha (float): Shape parameter (>0).

    Returns:
        np.ndarray: Tail probabilities.
    """
    t = np.asarray(t)
    tail_prob = np.zeros_like(t, dtype=np.float64)
    positive_mask = t > 0
    tail_prob[positive_mask] = np.exp(-t[positive_mask] ** (-alpha))
    tail_prob[~positive_mask] = 1.0
    return tail_prob