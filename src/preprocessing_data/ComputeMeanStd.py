from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import json
class Compute_mean_std(nn.Module):
    def __init__(self, batch_size: int = 256, num_channels: int = 3) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.mean = torch.zeros(self.num_channels)
        self.std = torch.zeros(self.num_channels)
    def forward(self, dataset: Dataset):
        self.data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        num_samples = 0
        for imgs, _ in self.data_loader:
            num_samples += imgs.shape[0]
            imgs = imgs.view(imgs.shape[0], self.num_channels, -1) # (B, C, -1)
            self.mean += torch.mean(imgs, dim=(0,2))*imgs.shape[0]
            self.std += torch.std(imgs, dim=(0,2))*imgs.shape[0]

        self.mean /= num_samples
        self.std /= num_samples

        return self.mean, self.std
    
    def Save_mean_std(self, save_dir: str) -> None:
        mean_std = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }
        with open(save_dir, "w") as f:
            json.dump(mean_std, f)
        print(f"Mean and std saved to {save_dir}")
    
    def Load_mean_std(self, load_dir: str) -> None:
        with open(load_dir, "r") as f:
            mean_std = json.load(f)
            self.mean = torch.tensor(mean_std["mean"])
            self.std = torch.tensor(mean_std["std"])
        print(f"Mean and std loaded from {load_dir}")
        return self.mean, self.std
