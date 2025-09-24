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

alpha = np.array([1., 2., 1., 5., 0.5, 10.])

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def montecarlo_conditional_prob(alpha_star: float, x: np.ndarray, t: float) -> float:
    """
    Estimate Monte Carlo conditional probability P(X0 > T | X1 > t).

    Args:
        alpha_star (float): Shape parameter for Frechet distribution.
        x (np.ndarray): Input data array of shape (N, d).
        t (float): Probability threshold.

    Returns:
        float: Estimated conditional probability.
    """
    q = - (1.0 / alpha_star) * np.log(-np.log(t))
    q = np.exp(q)
    mask = x[:, 1] > q
    if mask.sum() == 0:
        return 0.0
    joint = np.logical_and(x[:, 0] > q, mask).sum()
    return joint / mask.sum()


def fit_test(
    test_real_data: np.ndarray,
    gen_data: np.ndarray,
    sigma: float,
    q: np.ndarray,
    t_s: np.ndarray,
    log_scale: bool = True,
    seed: int = 50,
    name: str = None,           # name for the plot file
    output_file: str = None,     # file to save text output
    norm : bool = False,
    plain : bool = False
) -> None:
    """
    Evaluate generated data against real data using quantiles,
    Wasserstein distance, and conditional tail probabilities.
    Optionally saves the 2D KDE plot and writes results to a text file.

    Args:
        test_real_data (np.ndarray): Real dataset.
        gen_data (np.ndarray): Generated dataset to compare.
        sigma (float): Noise standard deviation.
        q (np.ndarray): Quantile levels to evaluate.
        t_s (np.ndarray): Tail probability thresholds for conditional evaluation.
        log_scale (bool): If True, data is assumed log-scaled; otherwise exponentiate with gamma.
        seed (int): Random seed for reproducibility.
        name (str, optional): If provided, saves the 2D KDE plot to this filename.
        output_file (str, optional): If provided, writes all text outputs to this file.
    """
    print("✅ Start testing for " + name)
    if seed is not None:
        set_seed(seed)

    if output_file is not None:
        f = open("results/"+output_file, "w")
        def print_func(*args, **kwargs):
            print(*args, **kwargs)  # sempre a schermo
            print(*args, **kwargs, file=f)  # anche nel file
    else:
        print_func = print

    if not isinstance(test_real_data, np.ndarray):
        test_real_data = np.array(test_real_data)
    if not isinstance(gen_data, np.ndarray):
        gen_data = np.array(gen_data)

    alpha_star = 4.0

    if not log_scale:
        gamma = alpha / alpha_star
        test_real_data = np.exp(gamma * test_real_data)
    if norm and not plain:
        test_real_data = (test_real_data - test_real_data.mean(axis=0)) / test_real_data.std(axis=0)

    noise = np.random.randn(*test_real_data.shape)
    frechet_data = test_real_data.copy()
    test_real_data = test_real_data + sigma * noise

    test_real_data = test_real_data[np.isfinite(test_real_data).all(axis=1)]
    gen_data = gen_data[np.isfinite(gen_data).all(axis=1)]
    print_func("Real data remaining:", test_real_data.shape[0])
    print_func("Generated data remaining:", gen_data.shape[0])

    gaussian = np.random.randn(*test_real_data.shape) * math.sqrt(1 + sigma**2)

    wass_gaussian = sliced_wasserstein_distance(
        test_real_data[:1_000_000, :], gaussian[:1_000_000, :], n_projections=20, p=2
    )
    wass_model = sliced_wasserstein_distance(
        test_real_data[:1_000_000, :], gen_data[:1_000_000, :], n_projections=20, p=2
    )

    wass_real = sliced_wasserstein_distance(
        test_real_data[:1_000_000, :],  test_real_data[1_000_000:2_000_000, :], n_projections=20, p=2
    )
    print_func("Wasserstein between Gaussian and real data:", wass_gaussian)
    print_func("Wasserstein between generated and real data:", wass_model)
    print_func("Wasserstein between real data and real data:", wass_real)

    test_real_data = test_real_data[:np.shape(gen_data)[0],:]

    for j in range(test_real_data.shape[1]):
        estimated_quantile = np.quantile(gen_data[:, j], q)
        real_quantile = np.quantile(test_real_data[:, j], q)
        quantile_p_inf = np.quantile(gaussian[:, j], q)

        print_func(f"\nAlpha = {alpha[j]}")
        print_func("Quantile | Real | Generated | Gaussian")
        for i, q_value in enumerate(q):
            print_func(f"{q_value:.6f} | {real_quantile[i]:15.4f} | {estimated_quantile[i]:15.4f} | {quantile_p_inf[i]:15.4f}")

    print_func("\nConditional tail probabilities for the first two marginals")
    print_func("t | threshold | Frechet | Noise-added | Gaussian | Generated")
    for t in t_s:
        tail_noise = montecarlo_conditional_prob(alpha_star, gaussian, t)
        tail_frechet = montecarlo_conditional_prob(alpha_star, frechet_data, t)
        tail_frechet_noise = montecarlo_conditional_prob(alpha_star, test_real_data, t)
        tail_gen = montecarlo_conditional_prob(alpha_star, gen_data, t)
        t_value = - (1.0 / alpha_star) * np.log(-np.log(t))
        print_func(f"{t:>8} | {t_value:.4f} | {tail_frechet:.4f} | {tail_frechet_noise:.4f} | {tail_noise:.4f} | {tail_gen:.4f}")

    datasets = {
        "Real data": test_real_data[:100_000, :],
        "Gaussian": gaussian[:100_000, :],
        "Generated": gen_data[:100_000, :]
    }

    marginals = [0, 1]

    def filter_data(data: np.ndarray, lower: float = -1e3, upper: float = 1e3) -> np.ndarray:
        mask = np.isfinite(data[:, marginals[0]]) & np.isfinite(data[:, marginals[1]])
        mask &= (data[:, marginals[0]] > lower) & (data[:, marginals[0]] < upper)
        mask &= (data[:, marginals[1]] > lower) & (data[:, marginals[1]] < upper)
        return data[mask]

    all_H = []
    for data in datasets.values():
        data_filtered = filter_data(data)
        H, _, _ = np.histogram2d(
            data_filtered[:, marginals[0]],
            data_filtered[:, marginals[1]],
            bins=100
        )
        all_H.append(H.ravel())

    all_H = np.concatenate(all_H)
    H_min, H_max = np.percentile(all_H, 1), np.percentile(all_H, 99)

    real_filtered = filter_data(datasets["Real data"])
    x_min, x_max = real_filtered[:, marginals[0]].min(), real_filtered[:, marginals[0]].max()
    y_min, y_max = real_filtered[:, marginals[1]].min(), real_filtered[:, marginals[1]].max()

    plt.figure(figsize=(15, 4))
    for i, (name_d, data) in enumerate(datasets.items(), 1):
        data_filtered = filter_data(data)

        H, _, _ = np.histogram2d(
            data_filtered[:, marginals[0]],
            data_filtered[:, marginals[1]],
            bins=100,
            range=[[x_min, x_max], [y_min, y_max]]
        )

        H = np.clip(H, H_min, H_max)

        plt.subplot(1, 3, i)
        plt.imshow(
            H.T,
            origin='lower',
            cmap='viridis',
            aspect='auto',
            extent=[x_min, x_max, y_min, y_max],
            vmin=H_min,
            vmax=H_max
        )

        plt.colorbar()
        plt.title(f"{name_d} - Marginals {marginals[0]} vs {marginals[1]}")
        plt.xlabel(f"X_{marginals[0]}")
        plt.ylabel(f"X_{marginals[1]}")

    plt.tight_layout()
    if name is not None:
        plt.savefig("results/" + name + "_plot.png", dpi=300)
        #print_func(f"Plot saved as {name}_plot.png")


    if output_file is not None:
        f.close()


# --- PARAMETERS ---

t_s = [0.9,0.99,0.999,0.9999,0.99999,0.999999, 0.9999999, 0.99999999, 0.999999999, 0.9999999999]
q = [0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
orig_dir = os.getcwd()
os.chdir("..")
sys.path.append(os.getcwd())
data_test = torch.load("config_SGM/data_frechet_experimentation.pt").to(device)
os.chdir(orig_dir)



gen_directory = "generation"
gen_dataset_names = [os.path.splitext(f)[0] for f in os.listdir(gen_directory) if f.endswith(".npy")]
print("Name datasets generated : ", gen_dataset_names)
gen_dataset_names = ['generation_noise']
for name in gen_dataset_names:
    logarithm = False
    sigma = 4.0
    norm = False
    plain = False
    if "log" in name:
        logarithm = True

    if "plain" in name:
        sigma = 0.
        plain = True
    else:
        sigma = 4.0

    if "norm" in name:
        norm = True
    output_file = name+".txt"

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    fit_test(test_real_data = data_test, gen_data = np.load("generation/"+name+".npy"),
        sigma = sigma,
        q = q,
        t_s = t_s,
        log_scale = logarithm,
        seed = 3000,
        name = name,
        output_file = output_file,
        norm = norm,
        plain = plain
        )
