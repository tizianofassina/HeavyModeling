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
set_seed(1000)
with open("config_SGM/config_max7.yaml", "r") as f:
    configuration = yaml.load(f, Loader=yaml.SafeLoader)


alpha = [3.,4.]
size_gen = 30_000_000

data_train_10e5 = torch.load("train_data/training_log_frechet_10e5.pt").detach().cpu().numpy().astype(np.float64)
data_train_10e6 = torch.load("train_data/training_log_frechet_10e6.pt").detach().cpu().numpy().astype(np.float64)
data_train_10e7 = torch.load("train_data/training_log_frechet_10e7.pt").detach().cpu().numpy().astype(np.float64)
data_train_10e5 = torch.tensor(np.exp(data_train_10e5)).to(device="cpu", dtype=dtype)
data_train_10e6 = torch.tensor(np.exp(data_train_10e6)).to(device="cpu", dtype=dtype)
data_train_10e7 = torch.tensor(np.exp(data_train_10e7)).to(device="cpu", dtype=dtype)

check_non_finite_torch(data_train_10e5)
check_non_finite_torch(data_train_10e6)
check_non_finite_torch(data_train_10e7)

sigma = configuration["diffusion_config"]["sigma_max"]

p_0 = torch.tensor(np.exp(log_frechet_rvs_copula_logistic(alpha = alpha, size = size_gen, theta = 3, device = "cpu").detach().cpu().numpy().astype(np.float64)), device = "cpu")
check_non_finite_torch(p_0, name = "p_0" )

z = torch.randn_like(p_0, device = "cpu", dtype=dtype)

p_T = p_0 + sigma * z
check_non_finite_torch(p_T, name = "p_T" )
p_inf = math.sqrt(sigma**2 + 1)*z
p_T_p_inf = torch.cat([p_T, p_inf], dim = 0).to(device="cpu", dtype=dtype)


log_dir = "runs"
model_dir = "models"
generation_dir = "generation"

os.makedirs(model_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
os.makedirs(generation_dir, exist_ok = True)

# --- TRAINING WITH 10^5 TRAINING POINTS ---

set_seed(1000)

name = "frechet_10e5train"
gen_big_10e5 = pipeline_diffusion(data = data_train_10e5, config_diffusion = configuration, log_dir = log_dir,model_dir = model_dir,name = name, sample = p_T_p_inf)
gen_p_T_10e5 = gen_big_10e5[:size_gen,:]
gen_p_inf_10e5 = gen_big_10e5[size_gen:,:]
np.save(generation_dir + "/gen_p_T_10e5", gen_p_T_10e5)
np.save(generation_dir + "/gen_p_inf_10e5", gen_p_inf_10e5)



# --- TRAINING WITH 10^6 TRAINING POINTS ---


set_seed(1000)

name = "frechet_10e6train"
gen_big_10e6 = pipeline_diffusion(data = data_train_10e6, config_diffusion = configuration, log_dir = log_dir,model_dir = model_dir,name = name, sample = p_T_p_inf)
gen_p_T_10e6 = gen_big_10e6[:size_gen,:]
gen_p_inf_10e6 = gen_big_10e6[size_gen:,:]
np.save(generation_dir +"/gen_p_T_10e6", gen_p_T_10e6)
np.save(generation_dir +"/gen_p_inf_10e6", gen_p_inf_10e6)

set_seed(1000)

name = "frechet_10e6train"
gen_big_10e7 = pipeline_diffusion(data = data_train_10e7, config_diffusion = configuration, log_dir = log_dir,model_dir = model_dir,name = name, sample = p_T_p_inf)
gen_p_T_10e7 = gen_big_10e7[:size_gen,:]
gen_p_inf_10e7 = gen_big_10e7[size_gen:,:]
np.save(generation_dir +"/gen_p_T_10e7", gen_p_T_10e7)
np.save(generation_dir +"/gen_p_inf_10e7", gen_p_inf_10e7)



