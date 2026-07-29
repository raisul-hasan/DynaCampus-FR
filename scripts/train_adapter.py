"""
Train Residual Domain Adapter — DynaCampus-FR Project
======================================================
Trains the PyTorch Residual MLP Adapter to bridge the domain gap between
low-quality surveillance probe embeddings and high-quality gallery prototypes.

Loss: Cosine Embedding Loss
Output: models/domain_adapter.pth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np

# Add project root to sys.path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapter.domain_adapter import ResidualDomainAdapter


def generate_synthetic_paired_data(num_samples: int = 1000, feature_dim: int = 512, noise_level: float = 0.25):
    """
    Generate paired (probe_surveillance, target_gallery) embeddings for domain adaptation training.
    Simulates resolution degradation, blur, and compression artifacts as directional noise.
    """
    # High-quality gallery prototypes (L2 normalized)
    targets = torch.randn(num_samples, feature_dim)
    targets = nn.functional.normalize(targets, p=2, dim=1)

    # Simulated surveillance probes with noise + blur vector shift
    noise = torch.randn(num_samples, feature_dim) * noise_level
    probes = targets + noise
    probes = nn.functional.normalize(probes, p=2, dim=1)

    return probes, targets


def train_domain_adapter(
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "models/domain_adapter.pth",
):
    """Train the Residual Domain Adapter."""
    print("[INFO] Initializing Residual Domain Adapter Training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # Generate or load paired dataset
    probes, targets = generate_synthetic_paired_data(num_samples=2000, feature_dim=512)
    dataset = TensorDataset(probes, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, optimizer
    model = ResidualDomainAdapter(feature_dim=512, hidden_dim=256).to(device)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    target_flag = torch.ones(batch_size).to(device)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_probes, batch_targets in dataloader:
            batch_probes = batch_probes.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            adapted = model(batch_probes)
            
            # Loss: 1 - cosine_similarity(adapted, target)
            flag = target_flag[: batch_probes.size(0)]
            loss = criterion(adapted, batch_targets, flag)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_probes.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch [{epoch:02d}/{epochs:02d}] - Cosine Loss: {epoch_loss:.5f}")

    # Save model weights
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Training complete! Saved weights to `{save_path}`")


if __name__ == "__main__":
    train_domain_adapter()
