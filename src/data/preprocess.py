# src/data/preprocess.py
"""
Fast, per-client data loading for heterogeneous FL simulation.
Loads pre-generated clients from data/processed_clients/{ct,mri}/clients.pkl
"""

import os
from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
LABEL_MAP = {"Healthy": 0, "Tumor": 1}


class BrainScanDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.filepaths = df["filepath"].values
        self.labels = df["label_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")  # Now directly RGB (no repeat later)
        except Exception as e:
            # Fallback: corrupted image → black image
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(train: bool = True):
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
        ] + base)
    return transforms.Compose(base)


def load_client_data(modality: str, client_id: int):
    """Load pre-split client DataFrame from simulate_clients.py"""
    repo_root = Path(__file__).resolve().parents[2]
    pkl_path = repo_root / "data" / "processed_clients" / modality / "clients.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Client data not found: {pkl_path}\n"
            "Run: python -m src.data.simulate_clients first!"
        )

    with open(pkl_path, "rb") as f:
        all_clients = pickle.load(f)

    if client_id >= len(all_clients):
        raise ValueError(f"Client {client_id} not found. Only {len(all_clients)} clients available for {modality}")

    return all_clients[client_id]


def create_dataloaders(client_id: int, modality: str):
    """
    Returns train_loader, val_loader, test_loader for a given client.
    Splits: 70% train, 15% val, 15% test (stratified)
    """
    df = load_client_data(modality, client_id)
    df = df.copy()

    # Stratified train/val/test split
    n = len(df)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    # Ensure at least 1 sample per class in each split
    try:
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, train_size=train_size, stratify=df["label_id"], random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=val_size, test_size=test_size,
                                           stratify=temp_df["label_id"], random_state=42)
    except:
        # Fallback if class missing
        train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=val_size, test_size=test_size, random_state=42)

    train_ds = BrainScanDataset(train_df, transform=get_transforms(train=True))
    val_ds = BrainScanDataset(val_df, transform=get_transforms(train=False))
    test_ds = BrainScanDataset(test_df, transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Client {client_id} ({modality.upper()}) | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test
    train_loader, val_loader, test_loader = create_dataloaders(client_id=0, modality="ct")
    print("DataLoader test passed!")