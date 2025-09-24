import os
import sys
import math
import random
import numpy as np
import torch
import lightning as L
import yaml
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

device = "cuda" if torch.cuda.is_available() else "cpu"
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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




try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

log_dir = os.path.join(base_dir, "runs")


# --- TRAINING DATA & PARAMETERS ---
orig_dir = os.getcwd()
os.chdir("..")
sys.path.append(os.getcwd())
from VE_SGM.pipeline_diffusion import pipeline_diffusion
from COMET.architecture import COMETFlow, MarginalLayer
from COMET.data_loader import CometDataModule, NoiseCometDataModule
from COMET.pipeline_comet import pipeline_comet
train_frechet = torch.load("train_data/training_frechet.pt").to(device)
train_uniform = torch.load("train_data/training_uniform.pt").to(device)
train_mix_gaussian = torch.load("train_data/training_mix_gaussian.pt").to(device)

with open("config_SGM/config_AR.yaml", "r") as f:
    config_AR = yaml.safe_load(f)

with open("config_SGM/config_student_AR.yaml", "r") as f:
    config_student_AR = yaml.safe_load(f)

with open("config_SGM/config_marginal.yaml", "r") as f:
    config_marginal = yaml.safe_load(f)

with open("config_SGM/config_student_marginal.yaml", "r") as f:
    config_student_marginal = yaml.safe_load(f)

with open("config_COMET/config_student.yaml", "r") as f:
    config_student = yaml.safe_load(f)

with open("config_COMET/config_noised_student.yaml", "r") as f:
    config_noised_student = yaml.safe_load(f)

with open("config_COMET/config_frechet.yaml", "r") as f:
    config_frechet = yaml.safe_load(f)

with open("config_COMET/config_noised_frechet.yaml", "r") as f:
    config_noised_frechet = yaml.safe_load(f)

with open("config_COMET/config_gaussian_mix.yaml", "r") as f:
    config_gaussian_mix = yaml.safe_load(f)

with open("config_COMET/config_noised_gaussian_mix.yaml", "r") as f:
    config_noised_gaussian_mix = yaml.safe_load(f)
os.chdir(orig_dir)


sigma_max = config_marginal["diffusion_config"]["sigma_max"]
alpha_star = 4.
alpha = torch.tensor([1.,2.,1.,5.,0.5, 10.], device = device)
gamma = alpha/alpha_star
gamma_np = gamma.detach().cpu().numpy()
generation_dir = "generation"
model_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(generation_dir, exist_ok=True)
"""

# --- TRAINING AND GENERATION - ANGLE/RAY APPROACH---
print()
print()

# Classic Frechet
set_seed(5000)
name = "frechet_SGM"
sigma_max = config_AR["diffusion_config"]["sigma_max"]
dim = train_frechet.shape[1] + 1
sample = torch.randn(config_AR["trainer_config"]["total_samples"], dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
rayon = torch.norm(torch.exp(train_frechet), dim=1, keepdim=True)
mid = dim // 2
train_frechet_normed = torch.exp(train_frechet) / rayon
data = torch.cat([train_frechet_normed[:, :mid], torch.log(rayon), train_frechet_normed[:, mid:]], dim=1)
generation = pipeline_diffusion(data, config_AR, log_dir, model_dir, name, sample)
name_array = name + "_array"
gen_left = generation[:, :mid]
r = np.exp(generation[:, mid:mid+1])
gen_right = generation[:, mid+1:]
generation = np.concatenate([gen_left, gen_right], axis=1)
generation = generation*r
check_non_finite(generation, name)
generation = np.sign(generation)*np.abs(generation)**gamma_np
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del data
del generation
del rayon
del sample

print()
print()

# Uniform mix student
set_seed(5000)
name = "uniform_mix_SGM"
sigma_max = config_student_AR["diffusion_config"]["sigma_max"]
dim = train_uniform.shape[1] + 1
sample = torch.randn(config_student_AR["trainer_config"]["total_samples"], dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
rayon = torch.norm(train_uniform, dim=1, keepdim=True)
mid = dim // 2
train_uniform_normed = train_uniform / rayon
data = torch.cat([train_uniform_normed[:, :mid], torch.log(rayon), train_uniform_normed[:, mid:]], dim=1)
generation = pipeline_diffusion(data, config_student_AR, log_dir, model_dir, name, sample)
name_array = name + "_array"
gen_left = generation[:, :mid]
r = np.exp(generation[:, mid:mid+1])
gen_right = generation[:, mid+1:]
generation = np.concatenate([gen_left, gen_right], axis=1)
generation = generation * r
check_non_finite(generation, name)
np.save(generation_dir + "/" +name_array + ".npy", generation)
del data
del generation
del rayon
del sample

print()
print()


# Gaussian mix and Frechet
set_seed(5000)
name = "gaussian_mix_SGM"
sigma_max = config_AR["diffusion_config"]["sigma_max"]
dim = train_mix_gaussian.shape[1] + 1
sample = torch.randn(config_AR["trainer_config"]["total_samples"], dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
rayon = torch.norm(train_mix_gaussian, dim=1, keepdim=True)
mid = dim // 2
train_mix_gaussian_normed = train_mix_gaussian / rayon
data = torch.cat([train_mix_gaussian_normed[:, :mid], torch.log(rayon), train_mix_gaussian_normed[:, mid:]], dim=1)
generation = pipeline_diffusion(data, config_AR, log_dir, model_dir, name, sample)
name_array = name + "_array"
gen_left = generation[:, :mid]
r = np.exp(generation[:, mid:mid+1])
gen_right = generation[:, mid+1:]
generation = np.concatenate([gen_left, gen_right], axis=1) * r
generation = np.sign(generation)*np.abs(generation)**gamma_np
check_non_finite(generation, name)
np.save(generation_dir + "/" +name_array + ".npy", generation)
del data
del generation
del rayon
del sample

print()
print()

# --- TRAINING AND GENERATION - MARGINAL LAYER APPROACH ---

# Classic Frechet
set_seed(5000)
name = "marginal_frechet_SGM"
data_frechet = torch.exp(train_frechet*gamma)
marginal_frechet = MarginalLayer(data_frechet, a = 0., b = 0.99, debug=False)
data_frechet_marginal,_ = marginal_frechet.forward(data_frechet)
dim = data_frechet.shape[1]
sample = torch.randn(config_marginal["trainer_config"]["total_samples"], dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
generation = pipeline_diffusion(data_frechet_marginal, config_marginal, log_dir, model_dir, name, sample)
generation,_ = marginal_frechet.forward(torch.tensor(generation, device = device), reverse=True)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del generation
del sample

print()
print()

# Uniform mix student
set_seed(5000)
name = "marginal_uniform_mix_SGM"
marginal_mix_uniform = MarginalLayer(train_uniform, a = 0.05, b = 0.95, debug=False)
data_uniform_mix_marginal,_ = marginal_mix_uniform.forward(train_uniform)
dim = train_uniform.shape[1]
sample = torch.randn(config_student_marginal["trainer_config"]["total_samples"], dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
generation = pipeline_diffusion(data_uniform_mix_marginal, config_student_marginal, log_dir, model_dir, name, sample)
generation,_ = marginal_mix_uniform.forward(torch.tensor(generation, device = device), reverse=True)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del generation
del sample


print()
print()

# Gaussian mix and Frechet
set_seed(5000)
name = "marginal_gaussian_mix_SGM"
data_gaussian_mix = torch.sign(train_mix_gaussian)*torch.abs(train_mix_gaussian)**gamma
marginal_mix_gaussian = MarginalLayer(data_gaussian_mix, a = 0.05, b = 0.95, debug=False)
data_gaussian_mix_marginal,_ = marginal_mix_gaussian.forward(data_gaussian_mix)
dim = data_gaussian_mix.shape[1]
sample = torch.randn(config_marginal["trainer_config"]["total_samples"], dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
generation = pipeline_diffusion(data_gaussian_mix_marginal,  config_marginal, log_dir, model_dir, name, sample)
generation,_ = marginal_mix_gaussian.forward(torch.tensor(generation, device = device), reverse=True)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del generation
del sample


print()
print()


# --- STANDARD COMET TRAINING, TO BE NOISED AFTER GENERATION ---


# Classic Frechet
set_seed(5000)
name = "frechet_COMET"
data = torch.exp(gamma * train_frechet)
generation = pipeline_comet(data, config_frechet, log_dir, model_dir, name = name)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del data
del generation

print()
print()

# Uniform mix student
set_seed(5000)
name = "uniform_mix_COMET"
data= train_uniform
generation = pipeline_comet(data, config_student, log_dir, model_dir, name = name)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del data
del generation

print()
print()

# Gaussian mix and Frechet
set_seed(5000)
name = "gaussian_mix_COMET"
data = torch.sign(train_mix_gaussian)*torch.abs(train_mix_gaussian)**gamma
generation = pipeline_comet(data, config_gaussian_mix, log_dir, model_dir, name = name)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del data
del generation



# --- DYNAMICALLY NOISED COMET TRAINING ---

print()
print()

# Classic Frechet
set_seed(5000)
name = "frechet_noised_COMET"
data = torch.exp(gamma * train_frechet)
sigma = config_noised_frechet["sigma"]
generation = pipeline_comet(data, config_noised_frechet, log_dir, model_dir, name = name)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del data
del generation

print()
print()
"""

# Uniform mix student
set_seed(5000)
name = "uniform_mix_COMET"
data= train_uniform
sigma = config_noised_student["sigma"]
generation = pipeline_comet(data , config_noised_student, log_dir, model_dir, name = name)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del data
del generation

print()
print()

# Gaussian mix and Frechet
set_seed(5000)
name = "gaussian_mix_COMET"
sigma = config_noised_gaussian_mix["sigma"]
data = torch.sign(train_mix_gaussian)*torch.abs(train_mix_gaussian)**gamma
generation = pipeline_comet(data + sigma*torch.randn_like(data), config_noised_gaussian_mix, log_dir, model_dir, name = name)
name_array = name + "_array"
check_non_finite(generation, name)
np.save(generation_dir + "/"+name_array + ".npy", generation)
del data
del generation





