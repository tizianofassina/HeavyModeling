import numpy as np
import torch
import lightning as L
from torch.utils.data import TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
from COMET.architecture import COMETFlow
from COMET.data_loader import NoiseCometDataModule, CometDataModule


def check_non_finite(arr: np.ndarray, name: str = ""):
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


def pipeline_comet(data: torch.Tensor, config_comet: dict, log_dir: str,
                   model_dir: str, name: str):
    """
    Full training and generation pipeline for COMETFlow.

    Args:
        data (torch.Tensor): Input training data.
        config_comet (dict): Configuration dictionary for COMET.
        log_dir (str): Directory to save TensorBoard logs.
        model_dir (str): Directory to save model checkpoints.
        generation_dir (str): Directory to save generated samples.
        name (str): Name used for logging and file naming.
    """
    # Extract configuration
    a = config_comet["a"]
    b = config_comet["b"]
    num_generation = config_comet["total_samples"]
    epochs = config_comet["max_epochs"]
    lr = config_comet["lr"]
    hidden_ds = config_comet["hidden_ds"]
    batch_gen = config_comet["batch_gen"]

    # Check data for non-finite values
    check_non_finite(data.cpu().numpy(), name=f"TRAINING {name}")

    # Initialize COMET model
    comet = COMETFlow(
        d=data.shape[1],
        hidden_ds=hidden_ds,
        lr=lr,
        data=data,
        a=a,
        b=b,
        conditional_noise=config_comet["conditional_noise"]
    )

    # Optional data normalization
    if config_comet["normalize_data"]:
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data = (data - data_mean) / data_std
        print("Data mean =", data_mean)
        print("Data std =", data_std)

    # Select appropriate DataModule
    if config_comet["noise_datamodule"]:
        data_module = NoiseCometDataModule(
            data=data,
            batch_size=config_comet["batch_size"],
            split=0.95,
            sigma=config_comet["sigma"],
            num_workers=config_comet["num_workers"],
            shuffle=False
        )
    else:
        data_module = CometDataModule(
            data=data,
            batch_size=config_comet["batch_size"],
            split=0.95,
            num_workers=config_comet["num_workers"],
            shuffle=False
        )

    # Setup trainer
    accelerator = "gpu" if data.device.type == "cuda" else "cpu"
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        logger=TensorBoardLogger(save_dir=log_dir, name=f"comet_{name}"),
        strategy=config_comet["strategy"]
    )

    # Train model
    trainer.fit(model=comet, datamodule=data_module)
    trainer.save_checkpoint(f"{model_dir}/comet_{name}.ckpt")

    # Evaluate and generate samples
    comet.eval()
    generation_list = []

    with torch.no_grad():
        for i, start in enumerate(range(0, num_generation, batch_gen), 1):
            end = min(start + batch_gen, num_generation)
            batch = torch.randn((end - start, data.shape[1]), device=data.device)
            gen = comet.forward(batch, reverse=True)[0]

            if config_comet["normalize_data"]:
                gen = gen * data_std + data_mean

            generation_list.append(gen.cpu())

            percent = (i) / (num_generation/batch_gen) * 100
            print(f"\r{name} : Generation progress {percent:.1f}% ", end="", flush=True)

        print()

    # Concatenate all generated batches
    generation = torch.cat(generation_list).detach().cpu().numpy()
    # Final check for non-finite values
    check_non_finite(generation, name)


    return generation