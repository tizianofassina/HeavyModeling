import lightning as L
import torch
import math
import numpy as np
from ot.sliced import sliced_wasserstein_distance
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import random
import sys
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

alpha = np.array([1., 2., 1., 0.5, 5.])


def check_non_finite(arr: np.ndarray, name: str = ""):
    """Check if a numpy array contains NaN or Inf values."""
    num_nan = np.isnan(arr).sum()
    num_pos_inf = np.isposinf(arr).sum()
    num_neg_inf = np.isneginf(arr).sum()
    num_inf = num_pos_inf + num_neg_inf
    if num_nan or num_inf:
        print(f"⚠️ Non-finite values detected in {name or 'array'}")
        print(f"NaNs: {num_nan}, +Inf: {num_pos_inf}, -Inf: {num_neg_inf}, Total Inf: {num_inf}")
    else:
        print(f"✅ No NaN/Inf found in {name or 'array'}")


def sample_gumbel_copula(size, dim, theta=3.0):
    a = 1.0 / theta

    # campioni uniformi e esponenziali in numpy float64
    U = np.random.rand(size) * math.pi
    W = np.random.exponential(1.0, size=size)

    numerator = np.sin(a * U)
    denominator = np.sin(U) ** (1.0 / a)
    factor = (np.sin((1 - a) * U) / W) ** ((1 - a) / a)

    S = (numerator / denominator) * factor
    S = S.reshape(size, 1)

    E = np.random.exponential(1.0, size=(size, dim))

    U = np.exp(-((E / S) ** a))

    return U.astype(np.float64)


def log_frechet_rvs_copula_logistic(alpha, size, theta=3.0):
    # alpha in numpy float64
    alpha_np = np.array(alpha, dtype=np.float64).reshape(1, -1)

    U = sample_gumbel_copula(size=size, dim=alpha_np.shape[1], theta=theta)
    eps = 1e-15
    U = np.clip(U, eps, 1 - eps)

    log_frechet_samples = - (1.0 / alpha_np) * np.log(-np.log(U))

    # check numerico in numpy
    nan_count = np.isnan(log_frechet_samples).sum()
    posinf_count = np.isposinf(log_frechet_samples).sum()
    neginf_count = np.isneginf(log_frechet_samples).sum()

    if nan_count > 0 or posinf_count > 0 or neginf_count > 0:
        print(f"[DEBUG] NaN: {nan_count}, +Inf: {posinf_count}, -Inf: {neginf_count}")
        raise ValueError("Numerical error: NaN or Inf detected in log-Frechet samples.")

    # converto solo alla fine in torch.float32 sul device scelto
    return log_frechet_samples

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def montecarlo_conditional_prob(alpha: float, x: np.ndarray, t: float) -> float:
    """
    Estimate Monte Carlo conditional probability P(X0 > T | X1 > t).

    Args:
        alpha_star (float): Shape parameter for Frechet distribution.
        x (np.ndarray): Input data array of shape (N, d).
        t (float): Probability threshold.

    Returns:
        float: Estimated conditional probability.
    """
    q_1 = - (1.0 / alpha[1]) * np.log(-np.log(t))
    q_0 = - (1.0 / alpha[0]) * np.log(-np.log(t))
    mask = x[:, 1] > q_1
    if mask.sum() == 0:
        return 0.0
    joint = np.logical_and(x[:, 0] > q_0, mask).sum()
    return joint / mask.sum()

# --- PARAMETERS ---


set_seed(10_000)

t_s = [0.9,0.99,0.999,0.9999,0.99999,0.999999, 0.9999999, 0.99999999, 0.999999999, 0.9999999999]
q = [0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
data_test = log_frechet_rvs_copula_logistic(alpha, size = 3_000_000, theta = 3.)



gen_directory = "data_prel_test"
gen_p_T = np.load(gen_directory + "/generated_original_5" + ".npy", allow_pickle=True)
gen_gaussian = np.load(gen_directory + "/generated_gaussian_5" + ".npy", allow_pickle=True)

check_non_finite(gen_p_T,  "generated_p_T")
check_non_finite(gen_gaussian,  "generated_gaussian")
check_non_finite(data_test,  "real data")


for j in range(np.shape(alpha)[0]):
    quantile_p_T = np.quantile(gen_p_T[:,j], q)
    quantile_gaussian = np.quantile(gen_gaussian[:,j], q)
    quantile_real = np.quantile(data_test[:,j], q)
    print(f"\nAlpha = {alpha[j]}")
    print("Quantile | Real | Real Est | p_T initialization | Gaussian initialization")
    for i, q_value in enumerate(q):
        quantile = - (1.0 / alpha[j]) * np.log(-np.log(q_value))
        print(f"{q_value:.6f} | {quantile:15.4f}|  {quantile_real[i]:15.4f} | {quantile_p_T[i]:15.4f} | {quantile_gaussian[i]:15.4f}")

print("\nConditional tail probabilities for the first two marginals")
print("t | threshold | Real Est | p_T initializiaton | Gaussian initialization")
for t in t_s:
    tail_p_T = montecarlo_conditional_prob(alpha, gen_p_T, t)
    tail_gaussian = montecarlo_conditional_prob(alpha, gen_gaussian, t)
    tail_real = montecarlo_conditional_prob(alpha, data_test, t)
    t_value = - (1.0 / alpha[1]) * np.log(-np.log(t))
    print(f"{t} | {t_value} | {tail_real} | {tail_p_T} | {tail_gaussian} ")


data_test = data_test[:6_000_000, :]


def estimate_swd(X, Y, n_runs=5, n_proj=50, p=2):
    vals = []
    for _ in range(n_runs):
        vals.append(sliced_wasserstein_distance(X, Y, n_projections=n_proj, p=p))
    return np.mean(vals), np.std(vals)



wass_gaussian_mean, wass_gaussian_std = estimate_swd(
        data_test[:1_000_000, :], gen_gaussian[:1_000_000, :], n_runs = 5, n_proj=20, p=2
    )
wass_p_T_mean, wass_p_T_std = estimate_swd(
        data_test[:1_000_000, :], gen_p_T[:1_000_000, :],  n_runs = 5, n_proj=20, p=2
    )

wass_real_mean, wass_real_std = estimate_swd(
        data_test[:1_000_000, :],  data_test[1_000_000:2_000_000, :], n_runs = 5, n_proj=20, p=2
    )


print("Wasserstein between Gaussian initialization data and real data: Mean ", wass_gaussian_mean, "Std ", wass_gaussian_std)
print("Wasserstein between p_T initialization data  and real data: Mean ", wass_p_T_mean, "Std ", wass_p_T_std)
print("Wasserstein between real data and real data: Mean ", wass_real_mean, "Std ", wass_real_std)




