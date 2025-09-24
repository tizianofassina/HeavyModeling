import numpy as np
import random
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset
import pandas as pd
import yaml
import torch
import math
from heavytail_tools.estimators_heavytail import hill_estimator
from VE_SGM.pipeline_diffusion import pipeline_diffusion
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


import os


if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    log_dir = os.path.join(base_dir, "runs")

    numpy_dir = os.path.join(base_dir, "generation")
    os.makedirs(numpy_dir, exist_ok=True)
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # --- BASIC PARAMETERS ---

    with open("config_SGM/config_trainer_gen.yaml", "r") as f:
        config_trainer_gen = yaml.safe_load(f)

    total_samples = config_trainer_gen["total_samples"]

    # --- DATA ---
    path = "config_SGM/data_frechet.pt"
    data = torch.load(path, map_location = device)
    dim = data.shape[1]

    # --- CONFIGURATION MAX10 ---
    with open("config_SGM/config_max10.yaml", "r") as f:
        config_diff = yaml.safe_load(f)

    # Log classic approach
    set_seed(42)
    name = "classic_log_max10"
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    name_array = name+"_array"
    np.save(name_array + ".npy", generation)

    # Log alternative approach
    set_seed(42)
    name = "alt_log_max10"
    data_alternative = data.clone()
    data_alternative[:, [3, 5]] = torch.exp(data_alternative[:, [3, 5]])
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_alternative, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation[:,[3,5]] = np.log(generation[:,[3,5]])
    name_array = name+"_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)

    # alpha_star = 5.
    set_seed(42)
    alpha_star = 5.
    name = "alpha_5_max10"
    data_5 = np.exp(data.cpu().numpy().astype(np.float64))
    gamma = np.min(hill_estimator(data_5, k=2000), alpha_star)/alpha_star
    data_5_torch = torch.from_numpy((data_5**gamma).astype(np.float32)).to(device)
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_5_torch, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation = np.log(generation)/gamma
    name_array = name+"_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)


    # alpha_star = 7.
    set_seed(42)
    alpha_star = 7.
    name = "alpha_7_max10"
    data_7 = np.exp(data.cpu().numpy().astype(np.float64))
    gamma = np.min(hill_estimator(data_7 , k = 2000), alpha_star)/alpha_star
    data_7_torch = torch.from_numpy((data_7**gamma).astype(np.float32)).to(device)
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_7_torch, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation = np.log(generation)/gamma
    name_array = name+"_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)

    # --- CONFIGURATION MAX12 ---
    with open("config_SGM/config_max12.yaml", "r") as f:
        config_diff = yaml.safe_load(f)

    # Log classic approach
    set_seed(42)
    name = "classic_log_max12"
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    name_array = name + "_array"
    np.save(name_array + ".npy", generation)

    # Log alternative approach
    set_seed(42)
    name = "alt_log_max12"
    data_alternative = data.clone()
    data_alternative[:, [3, 5]] = torch.exp(data_alternative[:, [3, 5]])
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_alternative, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation[:, [3, 5]] = np.log(generation[:, [3, 5]])
    name_array = name + "_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)

    # alpha_star = 5.
    set_seed(42)
    alpha_star = 5.
    name = "alpha_5_max12"
    data_5 = np.exp(data.cpu().numpy().astype(np.float64))
    gamma = np.min(hill_estimator(data_5, k=2000), alpha_star) / alpha_star
    data_5_torch = torch.from_numpy((data_5 ** gamma).astype(np.float32)).to(device)
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_5_torch, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation = np.log(generation) / gamma
    name_array = name + "_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)

    # alpha_star = 7.
    set_seed(42)
    alpha_star = 7.
    name = "alpha_7_max12"
    data_7 = np.exp(data.cpu().numpy().astype(np.float64))
    gamma = np.min(hill_estimator(data_7, k=2000), alpha_star) / alpha_star
    data_7_torch = torch.from_numpy((data_7 ** gamma).astype(np.float32)).to(device)
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_7_torch, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation = np.log(generation) / gamma
    name_array = name + "_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)

    # --- CONFIGURATION MAX15 ---
    with open("config_SGM/config_max15.yaml", "r") as f:
        config_diff = yaml.safe_load(f)


    # Log classic approach
    set_seed(42)
    name = "classic_log_max15"
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    name_array = name+"_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)

    # Log alternative approach
    set_seed(42)
    name = "alt_log_max15"
    data_alternative = data.clone()
    data_alternative[:, [3, 5]] = torch.exp(data_alternative[:, [3, 5]])
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_alternative, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation[:,[3,5]] = np.log(generation[:,[3,5]])
    name_array = name+"_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)

    # alpha_star = 5
    set_seed(42)
    alpha_star = 5.
    name = "alpha_5_max15"
    data_5 = np.exp(data.cpu().numpy().astype(np.float64))
    gamma = np.min(hill_estimator(data_5, k=2000), alpha_star)/alpha_star
    data_5_torch = torch.from_numpy((data_5**gamma).astype(np.float32)).to(device)
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_5_torch, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation = np.log(generation)/gamma
    name_array = name+"_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)

    # alpha_star = 7
    set_seed(42)
    alpha_star = 7.
    name = "alpha_7_max15"
    data_7 = np.exp(data.cpu().numpy().astype(np.float64))
    gamma = np.min(hill_estimator(data_7, k=2000 ), alpha_star)/alpha_star
    data_7_torch = torch.from_numpy((data_7**gamma).astype(np.float32)).to(device)
    sigma_max = config_diff["diffusion_config"]["sigma_max"]
    sample = torch.randn(total_samples, dim, device=device) * math.sqrt(sigma_max ** 2 + 1)
    generation = pipeline_diffusion(data_7_torch, config_trainer_gen, config_diff, log_dir, model_dir, name, sample)
    generation = np.log(generation)/gamma
    name_array = name+"_array"
    save_path = os.path.join(numpy_dir, name_array + ".npy")
    np.save(save_path, generation)






















