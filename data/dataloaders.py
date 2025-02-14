import torch
import os
import glob
from torch.utils.data import Dataset, DataLoader

class HARTimeSeriesDataset(Dataset):
    """Dataset for loading time-series individual samples"""

    def __init__(self, dataset_name, data_path):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.files = sorted(glob.glob(os.path.join(data_path, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])  # Load .pt file
        return sample  # Assuming each file contains (data, label)

def create_dataloader(args, split, batch_size):
    dataset_path = f"/home/tkimura4/data/datasets/{args.dataset}/time_individual_samples/"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found.")

    dataset = HARTimeSeriesDataset(args.dataset, dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
