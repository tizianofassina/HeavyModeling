import numpy as np
import heavytail_tools.estimators_heavytail
import os
from heavytail_tools.estimators_heavytail import quantile_estimator, log_frechet_quantile
import torch
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from config_SGM.data_training import log_frechet_rvs_copula_logistic as distribution_training
from heavytail_tools.estimators_heavytail import quantile_estimator
from heavytail_tools.estimators_heavytail import log_frechet_quantile as theoretical_quantile_function
from heavytail_tools.plot_functions import plot_quantiles_all_columns, plot_tailprob_all_columns, plot_condprob_all_columns
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


device = "cuda" if torch.cuda.is_avaliable() else "cpu"
set_seed(2000)

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
images_dir = os.path.join(base_dir, "images")
os.makedirs(images_dir, exist_ok=True)

quantile_dir = os.path.join(images_dir, "quantile")
os.makedirs(quantile_dir, exist_ok=True)

tails_dir = os.path.join(images_dir, "tail")
os.makedirs(tails_dir, exist_ok=True)

conditional_dir = os.path.join(images_dir, "conditional")
os.makedirs(conditional_dir, exist_ok=True)

alpha = [1., 2., 1., 5., 0.5, 10.]

# Load training data and comparaison data
training_data = torch.load("config_SGM/data_frechet.pt").cpu().numpy()
orig_data = distribution_training(alpha, size = 1_000_000, theta=3.0, device=device)

# Extract generated data
generation_dir = os.path.join(os.getcwd(), "generation")
file_names = [f for f in os.listdir(generation_dir) if os.path.isfile(os.path.join(generation_dir, f))]
file_names = [os.path.splitext(f)[0] for f in file_names if f.endswith(".npy")]
print(file_names)

alpha = [1., 2., 1., 5., 0.5, 10.]
q = [0.1,0.3,0.5,0.8,0.9,0.99,0.999,0.9999,0.99999,0.999999]
for name in file_names:
    generated_data = np.load(os.path.join("generation", f"{name}.npy"))

    del figs
    figs = plot_quantiles_all_columns(
            training_data = training_data,
            original_data = orig_data,
            generated_data = generated_data,
            theoretical_quantile_function = theoretical_quantile_function,
            alpha = alpha,
            q = q, title_prefix = name,
            n_stacks = 10)
    for i, fig in enumerate(figs):
        fig_dir = os.path.join(quantile_dir, f"q_{name}_{i}.png")
        fig.savefig(fig_dir)
        fig.close()


    del figs
    figs = plot_tailprob_all_columns(
        training_data=training_data,
        original_data=orig_data,
        generated_data=generated_data,
        alpha=alpha,
        theoretical_quantile_function=theoretical_quantile_function,
        q=q,
        title_prefix=name,
        n_stacks=10)
    for i, fig in enumerate(figs):
        fig_dir = os.path.join(tails_dir, f"t_{name}_{i}.png")
        fig.savefig(fig_dir)
        fig.close()

    del figs
    figs = plot_condprob_all_columns(
        training_data=training_data,
        original_data=orig_data,
        generated_data=generated_data,
        theoretical_quantile_function=theoretical_quantile_function,
        t=q, title_prefix=name,
        n_stacks=10)
    for i, fig in enumerate(figs):
        fig_dir = os.path.join(tails_dir, f"c_{name}_{i}.png")
        fig.savefig(fig_dir)
        fig.close()
    del generated_data






