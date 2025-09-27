import os
import lightning as L
import torch
import math
import numpy as np
from ot.sliced import sliced_wasserstein_distance
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import yaml
from train_data.data_training import log_frechet_rvs_copula_logistic
from VE_SGM.pipeline_diffusion import pipeline_diffusion
from heavytail_tools.general_tools import check_non_finite_torch, set_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print("Device : ", device)

set_seed(100)
with open("config_SGM/config_max7.yaml", "r") as f:
    configuration = yaml.load(f, Loader=yaml.SafeLoader)
configuration["denoiser_config"]["input_dim"] = 1

alpha = torch.tensor([5.], device = "cpu")

size_gen = 200_000
size_training = 100_000

data_train_10e5 = (-log_frechet_rvs_copula_logistic(alpha=alpha, size=size_gen, theta=3, device="cpu")).cpu().numpy()
data_train_10e5 = np.exp(data_train_10e5)
data_train_10e5 = torch.tensor(data_train_10e5, device="cpu", dtype=dtype)
d = data_train_10e5.shape[1]


check_non_finite_torch(data_train_10e5)

sigma = configuration["diffusion_config"]["sigma_max"]

p_0 = torch.tensor(np.exp(log_frechet_rvs_copula_logistic(alpha = alpha, size = size_gen, theta = 3, device = "cpu").detach().cpu().numpy().astype(np.float64)), device = "cpu")
check_non_finite_torch(p_0, name = "p_0" )

z = torch.randn_like(p_0, device = "cpu", dtype=dtype)

p_T = p_0 + sigma * z
check_non_finite_torch(p_T, name = "p_T" )

log_dir = "runs"
model_dir = "little_models"
generation_dir = "little_generation"

os.makedirs(model_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
os.makedirs(generation_dir, exist_ok = True)

set_seed(1000)

name = "little_frechet_10e5train"
x_generated = pipeline_diffusion(data = data_train_10e5, config_diffusion = configuration, log_dir = log_dir,model_dir = model_dir,name = name, sample = p_T)
x_generated = torch.tensor(x_generated, device=device, dtype=dtype)

print("Shape:", x_generated.shape)
print("Media campioni :", x_generated.mean(dim=0))
print("Std campioni :", x_generated.std(dim=0))
alpha = alpha.to(device)
# Riproduzione Frechet per confronto
U = torch.rand((size_gen, d), device=device)
frechet = (-torch.log(U)) ** (-1 / alpha)
U = torch.rand((size_gen, d), device=device)
frechet_2 = (-torch.log(U)) ** (-1 / alpha)
U = torch.rand((size_gen, d), device=device)
frechet_3 = (-torch.log(U)) ** (-1 / alpha)
U = torch.rand((size_gen, d), device=device)
frechet_4 = (-torch.log(U)) ** (-1 / alpha)

q = np.array([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])

quan_1 = np.quantile(frechet[:, 0].cpu().numpy(), q)
quan_2 = np.quantile(frechet_2[:, 0].cpu().numpy(), q)
quan_gen = np.quantile(x_generated[:, 0].cpu().numpy(), q)
print("Frechet_1 quantile", quan_1)
print("Frechet_2 quantile", quan_2)
print("Frechet_gen quantile", quan_gen)

print("Real Wass : ", sliced_wasserstein_distance(frechet.cpu().numpy(), frechet_2.cpu().numpy(), n_projections=10, p=2))
print("Real Wass : ", sliced_wasserstein_distance(frechet_2.cpu().numpy(), frechet_3.cpu().numpy(), n_projections=10, p=2))
print("Gen Wass : ", sliced_wasserstein_distance(x_generated.cpu().numpy(), frechet_4.cpu().numpy(), n_projections=10, p=2))

