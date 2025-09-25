import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def check_non_finite_np(arr: np.ndarray, name: str = ""):
    """
    Check if a numpy array contains NaN or Inf values and print a summary.

    Args:
        arr (np.ndarray): Array to check.
        name (str): Optional name for the array, used in print messages.
    """
    num_nan = np.isnan(arr).sum()
    num_pos_inf = np.isposinf(arr).sum()
    num_neg_inf = np.isneginf(arr).sum()
    num_inf = num_pos_inf + num_neg_inf

    if num_nan or num_inf:
        print(f"⚠️ Non-finite values detected in {name or 'array'}")
        print(f"NaNs: {num_nan}, +Inf: {num_pos_inf}, -Inf: {num_neg_inf}, Total Inf: {num_inf}")
    else:
        print(f"✅ No NaN/Inf found in {name or 'array'}")



def check_non_finite_torch(tensor: torch.Tensor, name: str = ""):
    """
    Check if a PyTorch tensor contains NaN or Inf values and print a summary.

    Args:
        tensor (torch.Tensor): Tensor to check.
        name (str): Optional name for the tensor, used in print messages.
    """
    num_nan = torch.isnan(tensor).sum().item()
    num_pos_inf = torch.isposinf(tensor).sum().item()
    num_neg_inf = torch.isneginf(tensor).sum().item()
    num_inf = num_pos_inf + num_neg_inf

    if num_nan or num_inf:
        print(f"⚠️ Non-finite values detected in {name or 'tensor'}")
        print(f"NaNs: {num_nan}, +Inf: {num_pos_inf}, -Inf: {num_neg_inf}, Total Inf: {num_inf}")
    else:
        print(f"✅ No NaN/Inf found in {name or 'tensor'}")

