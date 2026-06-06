#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Classifier2 on v5.7 pairwise data; save to Classifier2/artifacts_v5.7/.

Same training recipe as train_unified.py — only the CSV paths and the output
artifacts directory are different. The v1.8-trained model in artifacts/ is
preserved for direct comparison.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import train_unified as tu

tu.PAIRWISE_CSV = tu.CLASSIFIER2_ROOT / "Quantum Label Data - Pairwise_Comparisonv5.7.csv"
tu.ABSOLUTE_CSV = tu.CLASSIFIER2_ROOT / "Quantum Label Data - Absolute_Scoringv5.7.csv"

ARTIFACTS_DIR = tu.CLASSIFIER2_ROOT / "artifacts_v5.7"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(epochs=30, batch_size=16, lr=1e-4, samples_per_epoch=500):
    print("=" * 60)
    print("CLASSIFIER2 v5.7 TRAINING")
    print("=" * 60)
    print(f"Pairwise CSV: {tu.PAIRWISE_CSV.name}")
    print(f"Absolute CSV: {tu.ABSOLUTE_CSV.name}")
    print(f"Artifacts:    {ARTIFACTS_DIR}")
    sys.stdout.flush()

    device = pick_device()
    print(f"Device: {device}")
    sys.stdout.flush()

    train_pw, test_pw, ideal_images, bad_images, traj_images = tu.load_data()

    dataset = tu.UnifiedDataset(
        train_pw, ideal_images, bad_images, traj_images,
        transform=tu.get_transform(training=True),
        samples_per_epoch=samples_per_epoch,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = tu.BradleyTerryModel(pretrained_encoder_path=str(tu.PRETRAINED_ENCODER)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    sys.stdout.flush()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    history = []
    best_path = ARTIFACTS_DIR / "best_model.pth"
    final_path = ARTIFACTS_DIR / "final_model.pth"

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    sys.stdout.flush()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            loss = tu.compute_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / n_batches if n_batches else 0.0
        history.append({"epoch": epoch, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        sys.stdout.flush()

    torch.save(model.state_dict(), final_path)
    with open(ARTIFACTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Best model: {best_path}")
    print(f"Final model: {final_path}")

    print("\n" + "=" * 60)
    print("PAIRWISE HOLDOUT EVALUATION (v5.7)")
    print("=" * 60)
    tu.evaluate_pairwise_holdout(model, test_pw, device)

    return model


if __name__ == "__main__":
    train(epochs=30, batch_size=16, lr=1e-4, samples_per_epoch=500)
