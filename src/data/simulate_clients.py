# src/data/simulate_clients.py
"""
Generate realistic heterogeneous clients from raw CT/MRI datasets.
Supports:
- Dirichlet label distribution (non-IID)
- Variable client sizes (log-normal)
- Pure clients (100% one class)
- Config-driven (from configs/default.yaml)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import List
import yaml
import random


def load_config():
    config_path = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def simulate_heterogeneous_clients(
    modality: str,
    n_clients: int,
    dirichlet_alpha: float = 0.5,
    min_size: int = 80,
    max_size: int = 1800,
    seed: int = 42,
) -> List[pd.DataFrame]:
    np.random.seed(seed)
    random.seed(seed)

    # Resolve raw data path
    repo_root = Path(__file__).resolve().parents[2]
    if modality == "ct":
        raw_dir = repo_root / "data" / "raw" / "Dataset" / "Brain Tumor CT scan Images"
    elif modality == "mri":
        raw_dir = repo_root / "data" / "raw" / "Dataset" / "Brain Tumor MRI images"
    else:
        raise ValueError("modality must be 'ct' or 'mri'")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_dir}")

    # Collect all images
    data = []
    for label_name, label_id in [("Healthy", 0), ("Tumor", 1)]:
        label_dir = raw_dir / label_name
        if not label_dir.exists():
            continue
        for img_path in label_dir.glob("*.*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                data.append({
                    "filepath": str(img_path),
                    "label": label_name,
                    "label_id": label_id,
                })

    df = pd.DataFrame(data)
    print(f"[{modality.upper()}] Total images: {len(df)} | Healthy: {(df.label_id == 0).sum()} | Tumor: {(df.label_id == 1).sum()}")

    n_healthy = (df.label_id == 0).sum()
    n_tumor = (df.label_id == 1).sum()

    # Dirichlet allocation for label proportions
    proportions = np.random.dirichlet(alpha=[dirichlet_alpha] * n_clients)
    proportions = proportions / proportions.sum()  # renormalize

    client_dfs = []
    healthy_df = df[df.label_id == 0].copy()
    tumor_df = df[df.label_id == 1].copy()

    for client_id in range(n_clients):
        # Size: log-normal + clipped
        raw_size = int(np.random.lognormal(mean=5.3, sigma=0.9))
        target_size = np.clip(raw_size, min_size, max_size)

        # Label proportion from Dirichlet
        tumor_ratio = proportions[client_id]
        n_tumor_target = int(target_size * tumor_ratio)
        n_healthy_target = target_size - n_tumor_target

        # Sample without replacement
        sample_tumor = tumor_df.sample(n=min(n_tumor_target, len(tumor_df)), replace=False) if n_tumor_target > 0 and len(tumor_df) > 0 else pd.DataFrame()
        sample_healthy = healthy_df.sample(n=min(n_healthy_target, len(healthy_df)), replace=False) if n_healthy_target > 0 and len(healthy_df) > 0 else pd.DataFrame()

        client_data = pd.concat([sample_tumor, sample_healthy], ignore_index=True)

        # Fallback: if not enough data → random sample
        if len(client_data) < target_size // 2 and len(df) > 0:
            client_data = df.sample(n=target_size, replace=False)

        # Remove used images
        if len(client_data) > 0:
            used_idx = client_data.index
            healthy_df = healthy_df.drop(healthy_df.index.intersection(used_idx), errors="ignore")
            tumor_df = tumor_df.drop(tumor_df.index.intersection(used_idx), errors="ignore")

        client_data = client_data.sample(frac=1.0, random_state=seed + client_id).reset_index(drop=True)
        client_data["client_id"] = client_id
        client_data["modality"] = modality

        client_dfs.append(client_data)

        tumor_pct = client_data["label_id"].mean() * 100
        print(f"  Client {client_id:2d} | Size: {len(client_data):4d} | Tumor: {tumor_pct:5.1f}%")

    return client_dfs


def main():
    config = load_config()
    data_cfg = config.get("data", {})
    fl_cfg = config.get("fl", {})

    n_clients = fl_cfg.get("num_clients", 60)
    alpha = data_cfg.get("dirichlet_alpha", 0.3)
    partition = data_cfg.get("partition", "dirichlet")

    if data_cfg.get("iid", False):
        print("IID mode requested — but this script uses Dirichlet (non-IID). Set iid=False.")
        return

    for modality in ["ct", "mri"]:
        print(f"\nGenerating {n_clients//2} clients for {modality.upper()} (Dirichlet α={alpha})...")
        clients = simulate_heterogeneous_clients(
            modality=modality,
            n_clients=n_clients // 2,
            dirichlet_alpha=alpha,
            seed=42,
        )

        out_dir = Path("data") / "processed_clients" / modality
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "clients.pkl"

        with open(out_path, "wb") as f:
            pickle.dump(clients, f)

        print(f"Saved {len(clients)} clients → {out_path}\n")


if __name__ == "__main__":
    main()