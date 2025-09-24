import numpy as np
import heavytail_tools.estimators_heavytail
import os
from heavytail_tools.estimators_heavytail import quantile_estimator, log_frechet_quantile, tail_probability_mc
import torch
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt



def plot_quantiles_all_columns(
    original_data: np.ndarray,
    training_data: np.ndarray,
    generated_data: np.ndarray,
    theoretical_quantile_function: callable,
    alpha : np.ndarray,
    q: Union[float, np.ndarray],
    title_prefix: str,
    n_stacks: int
) -> List[plt.Figure]:
    """
    Plots the quantiles for all columns of the data, comparing original, generated, and theoretical values.

    Args:
        original_data (np.ndarray): Original data of shape (N, m).
        training_data (np.ndarray): Training data of shape (training_size, m).
        generated_data (np.ndarray): Generated data of shape (N, m).
        theoretical_quantile_function (callable): Function returning the theoretical quantile given q.
        q (float or np.ndarray): Quantile(s) to estimate (0 < q < 1).
        title_prefix (str): Prefix for the plot titles.
        n_stacks (int): Number of stacks to compute mean and standard deviation.

    Returns:
        List[plt.Figure]: List of matplotlib figure objects, one per column.
    """
    m = original_data.shape[1]
    figs = []

    for col in range(m):
        th = theoretical_quantile_function(alpha[col], q).flatten()

        N_gen = generated_data.shape[0]
        N_orig = original_data.shape[0]
        stack_size_gen = N_gen // n_stacks
        stack_size_orig = N_orig // n_stacks
        quantiles_orig = []
        quantiles_gen = []

        for i in range(n_stacks):
            # Generated data
            start_gen = i * stack_size_gen
            end_gen = (i + 1) * stack_size_gen if i < n_stacks - 1 else N_gen
            stack_gen = generated_data[start_gen:end_gen, col]
            stack_quantile_gen = np.quantile(stack_gen, q)
            quantiles_gen.append(stack_quantile_gen)

            # Original data
            start_or = i * stack_size_orig
            end_or = (i + 1) * stack_size_orig if i < n_stacks - 1 else N_orig
            stack_or = original_data[start_or:end_or, col]
            stack_quantile_or = np.quantile(stack_or, q)
            quantiles_orig.append(stack_quantile_or)

        quantiles_gen = np.stack(quantiles_gen, axis=0)
        quantiles_orig = np.stack(quantiles_orig, axis=0)

        mean_gen = quantiles_gen.mean(axis=0)
        std_gen = quantiles_gen.std(axis=0)
        mean_orig = quantiles_orig.mean(axis=0)
        std_orig = quantiles_orig.std(axis=0)

        quantile_training = np.quantile(training_data[col, :], q)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(th, mean_orig, 'k-', label='Original')
        ax.fill_between(th, mean_orig - std_orig, mean_orig + std_orig, color='gray', alpha=0.3)
        ax.plot(th, mean_gen, 'r-', label='Generated')
        ax.fill_between(th, mean_gen - std_gen, mean_gen + std_gen, color='red', alpha=0.3)
        ax.plot(th, th, 'b--', label='Theoretical')
        ax.plot(th, quantile_training, 'g-', label='Training')
        ax.set_xlabel("Theoretical Quantile")
        ax.set_ylabel(f"{q}-Quantile")
        ax.set_title(f"{title_prefix} - Column {col}")
        ax.legend()

        figs.append(fig)

    return figs



def plot_tailprob_all_columns(
    original_data: np.ndarray,
    training_data: np.ndarray,
    generated_data: np.ndarray,
    theoretical_quantile_function: callable,
    alpha : np.ndarray,
    t: Union[float, np.ndarray],
    title_prefix: str,
    n_stacks: int
) -> List[plt.Figure]:
    """
    Plots the tail probabilities P(X > F^{-1}(t)) for all columns,
    comparing original, generated, and theoretical values.

    Args:
        original_data (np.ndarray): Original data of shape (N, m).
        training_data (np.ndarray): Training data of shape (training_size, m).
        generated_data (np.ndarray): Generated data of shape (N, m).
        theoretical_quantile_function (callable): Function returning the theoretical quantile given t.
        t (float or np.ndarray): Probabilities (0 < t < 1).
        title_prefix (str): Prefix for the plot titles.
        n_stacks (int): Number of stacks to compute mean and standard deviation.

    Returns:
        List[plt.Figure]: List of matplotlib figure objects, one per column.
    """
    m = original_data.shape[1]
    figs = []

    # threshold corrispondente: F^{-1}(t)


    for col in range(m):
        th = theoretical_quantile_function(alpha[col], t).flatten()
        N_gen = generated_data.shape[0]
        N_orig = original_data.shape[0]
        stack_size_gen = N_gen // n_stacks
        stack_size_orig = N_orig // n_stacks
        tails_orig = []
        tails_gen = []

        for i in range(n_stacks):
            # Generated data
            start_gen = i * stack_size_gen
            end_gen = (i + 1) * stack_size_gen if i < n_stacks - 1 else N_gen
            stack_gen = generated_data[start_gen:end_gen, col]
            stack_tail_gen = [(stack_gen > thr).mean() for thr in th]
            tails_gen.append(stack_tail_gen)

            # Original data
            start_or = i * stack_size_orig
            end_or = (i + 1) * stack_size_orig if i < n_stacks - 1 else N_orig
            stack_or = original_data[start_or:end_or, col]
            stack_tail_or = [(stack_or > thr).mean() for thr in th]
            tails_orig.append(stack_tail_or)

        tails_gen = np.stack(tails_gen, axis=0)
        tails_orig = np.stack(tails_orig, axis=0)

        mean_gen = tails_gen.mean(axis=0)
        std_gen = tails_gen.std(axis=0)
        mean_orig = tails_orig.mean(axis=0)
        std_orig = tails_orig.std(axis=0)

        # Training tail probability
        training_tail = [(training_data[:, col] > thr).mean() for thr in th]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(th, mean_orig, 'k-', label='Original')
        ax.fill_between(th, mean_orig - std_orig, mean_orig + std_orig, color='gray', alpha=0.3)
        ax.plot(th, mean_gen, 'r-', label='Generated')
        ax.fill_between(th, mean_gen - std_gen, mean_gen + std_gen, color='red', alpha=0.3)
        ax.plot(th, 1 - t, 'b--', label='Theoretical')  # teorico: P(X>F^{-1}(t)) = 1 - t
        ax.plot(th, training_tail, 'g-', label='Training')

        ax.set_xlabel("Threshold (F^{-1}(t))")
        ax.set_ylabel("Tail Probability P(X > threshold)")
        ax.set_title(f"{title_prefix} - Column {col}")
        ax.legend()

        figs.append(fig)

    return figs


from heavytail_tools.estimators_heavytail import conditional_tail_prob_frechet

def plot_condprob_all_columns(
    original_data: np.ndarray,
    training_data: np.ndarray,
    generated_data: np.ndarray,
    theoretical_data: np.ndarray,
    theoretical_quantile_function : callable,
    t: Union[float, np.ndarray],
    title_prefix: str,
    n_stacks: int
) -> List[plt.Figure]:


    m = original_data.shape[1]
    figs = []

    # threshold corrispondente: F^{-1}(t)
    th = theoretical_quantile_function(t).flatten()

    for col in range(m):
        N_gen = generated_data.shape[0]
        N_orig = original_data.shape[0]
        stack_size_gen = N_gen // n_stacks
        stack_size_orig = N_orig // n_stacks
        tails_orig = []
        tails_gen = []

        for i in range(n_stacks):
            # Generated data
            start_gen = i * stack_size_gen
            end_gen = (i + 1) * stack_size_gen if i < n_stacks - 1 else N_gen
            stack_gen = generated_data[start_gen:end_gen, col]
            stack_tail_gen = tail_probability_mc(stack_gen, th)
            tails_gen.append(stack_tail_gen)

            # Original data
            start_or = i * stack_size_orig
            end_or = (i + 1) * stack_size_orig if i < n_stacks - 1 else N_orig
            stack_or = original_data[start_or:end_or, col]
            stack_tail_or = [(stack_or > thr).mean() for thr in th]
            tails_orig.append(stack_tail_or)

        tails_gen = np.stack(tails_gen, axis=0)
        tails_orig = np.stack(tails_orig, axis=0)

        mean_gen = tails_gen.mean(axis=0)
        std_gen = tails_gen.std(axis=0)
        mean_orig = tails_orig.mean(axis=0)
        std_orig = tails_orig.std(axis=0)

        # Training tail probability
        training_tail = [(training_data[:, col] > thr).mean() for thr in th]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(th, mean_orig, 'k-', label='Original')
        ax.fill_between(th, mean_orig - std_orig, mean_orig + std_orig, color='gray', alpha=0.3)
        ax.plot(th, mean_gen, 'r-', label='Generated')
        ax.fill_between(th, mean_gen - std_gen, mean_gen + std_gen, color='red', alpha=0.3)
        ax.plot(th, 1 - t, 'b--', label='Theoretical')  # teorico: P(X>F^{-1}(t)) = 1 - t
        ax.plot(th, training_tail, 'g-', label='Training')

        ax.set_xlabel("Threshold (F^{-1}(t))")
        ax.set_ylabel("Tail Probability P(X > threshold)")
        ax.set_title(f"{title_prefix} - Column {col}")
        ax.legend()

        figs.append(fig)

    return figs
