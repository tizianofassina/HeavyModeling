import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import lightning as L


def add_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise to a tensor."""
    return x + torch.randn_like(x) * sigma


class NoiseDataset(Dataset):
    """Dataset wrapper that applies dynamic Gaussian noise to a base dataset."""
    def __init__(self, base_dataset: Dataset, sigma: float):
        if isinstance(base_dataset, torch.Tensor):
            base_dataset = TensorDataset(base_dataset)
        self.base_dataset = base_dataset
        self.sigma = sigma

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        x = self.base_dataset[index]
        if isinstance(x, tuple):
            # add noise only to the first element, keep rest intact
            x = (add_noise(x[0], self.sigma),) + x[1:]
        else:
            x = add_noise(x, self.sigma)
        return x

class CometDataModule(L.LightningDataModule):
    """DataModule for raw tensors or datasets without noise."""
    def __init__(self, data, batch_size=32, split=0.95, num_workers=0, shuffle=False):
        super().__init__()
        if isinstance(data, torch.Tensor):
            data = TensorDataset(data)
        self.data = data
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage=None):
        n_total = len(self.data)
        n_train = int(self.split * n_total)
        n_val = n_total - n_train
        self.train_dataset, self.val_dataset = random_split(self.data, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )


class NoiseCometDataModule(L.LightningDataModule):
    """DataModule for dynamically noised tensors or datasets with train/validation split."""

    def __init__(self, data, batch_size=32, sigma=0.0, split=0.95, num_workers=0, shuffle= False):
        """
        Args:
            data (torch.Tensor or Dataset): Input data.
            batch_size (int): Batch size for training/validation.
            sigma (float): Standard deviation of Gaussian noise applied dynamically.
            split (float): Fraction of data to use for training (rest is validation).
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        if isinstance(data, torch.Tensor):
            data = TensorDataset(data)
        self.base_dataset = data
        self.batch_size = batch_size
        self.sigma = sigma
        self.split = split
        self.num_workers = num_workers
        self.use_pin_memory = torch.cuda.is_available()
        self.shuffle = shuffle


    def setup(self, stage=None):
        """Apply noise dynamically and split dataset into train/validation."""
        noisy_dataset = NoiseDataset(self.base_dataset, self.sigma)
        n_train = int(self.split * len(noisy_dataset))
        n_val = len(noisy_dataset) - n_train
        self.train_dataset, self.val_dataset = random_split(
            noisy_dataset,
            lengths=[n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.use_pin_memory,
            prefetch_factor=None if self.num_workers == 0 else 2,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.use_pin_memory,
            prefetch_factor=None if self.num_workers == 0 else 2,
            persistent_workers=self.num_workers > 0,
        )