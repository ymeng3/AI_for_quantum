#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Training Script for Classifier2 (v2.1)

Combines multiple data sources:
1. Pairwise comparison data (Bradley-Terry loss) with confidence weighting
2. Ideal reference images for ALL 5 reconstruction types (anchor loss)
3. Bad images (quality filtering loss)

Current data (January 2025):
- Pairwise comparisons: 678 (from v1.8)
- Ideal HTR: 21 images
- Ideal RT13: 21 images
- Ideal 1x1: 27 images
- Ideal c6x2: 24 images
- Ideal Twinned: 6 images
- Bad images: 18
- Total ideal images: 99

Usage:
    python train_unified.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"
CLASSIFIER2_ROOT = Path(__file__).parent
CLASSIFIER1_ROOT = PROJECT_ROOT / "Classifier1"

# CSV files (v1.8)
PAIRWISE_CSV = CLASSIFIER2_ROOT / "Quantum Label Data - Pairwise_Comparisonv1.8.csv"
ABSOLUTE_CSV = CLASSIFIER2_ROOT / "Quantum Label Data - Absolute_Scoringv1.8.csv"

# Ideal image directories (all 5 types)
IDEAL_DIRS = {
    'HTR': DATA_ROOT / "STO_ideal_HTR",
    '(√13 x √13)': DATA_ROOT / "STO_ideal_RT13",
    '(1 x 1)': DATA_ROOT / "STO_ideal_1x1",
    'Twinned(2 x 1)': DATA_ROOT / "STO_ideal_Twinned2x1",
    'c(6 x 2)': DATA_ROOT / "STO_ideal_c6x2",
}

# Legacy aliases for backward compatibility
IDEAL_HTR_DIR = IDEAL_DIRS['HTR']
IDEAL_RT13_DIR = IDEAL_DIRS['(√13 x √13)']

# Trajectory directory
TRAJECTORY_DIR = DATA_ROOT / "Trajectories"

# Pretrained encoder from Classifier1
PRETRAINED_ENCODER = CLASSIFIER1_ROOT / "artifacts" / "encoders" / "simclr_resnet18_encoder.pth"

# Reconstruction types
RECONSTRUCTION_TYPES = [
    '(1 x 1)',
    'Twinned(2 x 1)',
    'c(6 x 2)',
    '(√13 x √13)',
    'HTR'
]
NUM_TYPES = len(RECONSTRUCTION_TYPES)
TYPE_TO_IDX = {t: i for i, t in enumerate(RECONSTRUCTION_TYPES)}

# Winner mapping
WINNER_MAP = {'1': 0, '2': 1, 'tie': 2, 'not_apply': 3}

# Confidence mapping
CONFIDENCE_MAP = {
    'Confident': 1.0,
    'Somewhat sure': 0.7,
    '': 1.0,  # Default
    None: 1.0,
}


def get_transform(training=True):
    """Get image transforms for training/eval."""
    if training:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])


class UnifiedDataset(Dataset):
    """
    Dataset that combines pairwise, ideal, and bad image data.

    Each sample is a pair of images with associated labels.
    """

    def __init__(self, pairwise_df, ideal_images, bad_images, trajectory_images,
                 transform=None, samples_per_epoch=500):
        self.pairwise_df = pairwise_df
        self.ideal_images = ideal_images  # {'HTR': [...], '(√13 x √13)': [...], ...}
        self.bad_images = bad_images  # list of paths
        self.trajectory_images = trajectory_images  # list of all trajectory images
        self.transform = transform or get_transform(training=True)
        self.samples_per_epoch = samples_per_epoch

        # Pre-filter valid pairwise entries
        self.valid_pairwise = []
        for idx in range(len(pairwise_df)):
            row = pairwise_df.iloc[idx]
            rt = row['Reconstruction_Type']
            winner = str(row['Winner'])
            if rt in TYPE_TO_IDX and winner in WINNER_MAP:
                self.valid_pairwise.append(idx)

        # Count ideal images by type
        total_ideal = sum(len(imgs) for imgs in ideal_images.values())

        print(f"Valid pairwise entries: {len(self.valid_pairwise)}")
        print(f"Total ideal images: {total_ideal}")
        for rt, imgs in ideal_images.items():
            if imgs:
                print(f"  {rt}: {len(imgs)}")
        print(f"Bad images: {len(bad_images)}")
        print(f"Trajectory images: {len(trajectory_images)}")

    def __len__(self):
        return self.samples_per_epoch

    def _load_image(self, path):
        """Load and transform an image."""
        path = Path(path)
        if not path.exists():
            # Try relative to DATA_ROOT
            path = DATA_ROOT / path
        img = Image.open(path).convert('L')
        return self.transform(img)

    def __getitem__(self, idx):
        # Randomly choose data source type
        # 60% pairwise, 25% ideal anchoring, 15% bad image
        r = random.random()

        total_ideal = sum(len(imgs) for imgs in self.ideal_images.values())

        if r < 0.60 and len(self.valid_pairwise) > 0:
            return self._get_pairwise_sample()
        elif r < 0.85 and total_ideal > 0:
            return self._get_ideal_sample()
        elif len(self.bad_images) > 0:
            return self._get_bad_sample()
        else:
            return self._get_pairwise_sample()

    def _get_pairwise_sample(self):
        """Get a pairwise comparison sample with confidence weighting."""
        idx = random.choice(self.valid_pairwise)
        row = self.pairwise_df.iloc[idx]

        path1 = row['Image1_Path']
        path2 = row['Image2_Path']
        rt = row['Reconstruction_Type']
        winner = str(row['Winner'])

        # Get confidence weight
        confidence = row.get('Confidence', None)
        weight = CONFIDENCE_MAP.get(confidence, 1.0) if pd.notna(confidence) else 1.0

        # Random flip (data augmentation)
        if random.random() < 0.5:
            path1, path2 = path2, path1
            if winner == '1':
                winner = '2'
            elif winner == '2':
                winner = '1'

        img1 = self._load_image(DATA_ROOT / path1)
        img2 = self._load_image(DATA_ROOT / path2)

        return {
            'type': 'pairwise',
            'image1': img1,
            'image2': img2,
            'reconstruction_type': TYPE_TO_IDX[rt],
            'winner': WINNER_MAP[winner],
            'weight': weight
        }

    def _get_ideal_sample(self):
        """Get an ideal anchoring sample (ideal vs random trajectory or ideal vs other-type ideal)."""
        # Choose a reconstruction type that has ideal images
        available_types = [t for t, imgs in self.ideal_images.items() if imgs]
        if not available_types:
            return self._get_pairwise_sample()

        ideal_type = random.choice(available_types)
        type_idx = TYPE_TO_IDX[ideal_type]
        ideal_path = random.choice(self.ideal_images[ideal_type])

        # 70% ideal vs trajectory, 30% ideal vs other-type ideal
        if random.random() < 0.7:
            # Ideal vs random trajectory
            other_path = random.choice(self.trajectory_images)
        else:
            # Ideal vs other-type ideal (cross-type anchoring)
            other_types = [t for t in available_types if t != ideal_type]
            if other_types:
                other_type = random.choice(other_types)
                other_path = random.choice(self.ideal_images[other_type])
            else:
                other_path = random.choice(self.trajectory_images)

        img_ideal = self._load_image(ideal_path)
        img_other = self._load_image(other_path)

        # Ideal should win for its type
        # Random order
        if random.random() < 0.5:
            return {
                'type': 'ideal',
                'image1': img_ideal,
                'image2': img_other,
                'reconstruction_type': type_idx,
                'winner': 0,  # image1 (ideal) wins
                'weight': 1.0
            }
        else:
            return {
                'type': 'ideal',
                'image1': img_other,
                'image2': img_ideal,
                'reconstruction_type': type_idx,
                'winner': 1,  # image2 (ideal) wins
                'weight': 1.0
            }

    def _get_bad_sample(self):
        """Get a bad image sample (bad vs random trajectory)."""
        bad_path = random.choice(self.bad_images)
        traj_path = random.choice(self.trajectory_images)

        img_bad = self._load_image(bad_path)
        img_traj = self._load_image(traj_path)

        # For bad images, we use a special marker
        # The trajectory should win (bad loses)
        type_idx = random.randint(0, NUM_TYPES - 1)  # Any type

        if random.random() < 0.5:
            return {
                'type': 'bad',
                'image1': img_bad,
                'image2': img_traj,
                'reconstruction_type': type_idx,
                'winner': 1,  # image2 (trajectory) wins over bad
                'weight': 1.0
            }
        else:
            return {
                'type': 'bad',
                'image1': img_traj,
                'image2': img_bad,
                'reconstruction_type': type_idx,
                'winner': 0,  # image1 (trajectory) wins over bad
                'weight': 1.0
            }


class BradleyTerryModel(nn.Module):
    """Bradley-Terry reward model."""

    def __init__(self, pretrained_encoder_path=None, hidden_dim=256):
        super().__init__()

        # ResNet18 encoder
        backbone = models.resnet18(pretrained=False)
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)

        # Load pretrained weights
        if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
            print(f"Loading pretrained encoder from: {pretrained_encoder_path}")
            state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict)

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_TYPES)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze(-1).squeeze(-1)
        return self.reward_head(h)


def compute_loss(model, batch, device):
    """Compute combined loss for a batch with confidence weighting."""
    img1 = batch['image1'].to(device)
    img2 = batch['image2'].to(device)
    type_idx = batch['reconstruction_type'].to(device)
    winner = batch['winner'].to(device)
    weight = batch['weight'].float().to(device)

    # Get rewards
    r1 = model(img1)  # [B, NUM_TYPES]
    r2 = model(img2)  # [B, NUM_TYPES]

    batch_size = img1.shape[0]
    r1_type = r1[torch.arange(batch_size, device=device), type_idx]
    r2_type = r2[torch.arange(batch_size, device=device), type_idx]

    loss = torch.tensor(0.0, device=device)

    # image1 wins: maximize P(r1 > r2)
    mask_1 = (winner == 0)
    if mask_1.any():
        l = (-F.logsigmoid(r1_type[mask_1] - r2_type[mask_1])) * weight[mask_1]
        loss = loss + l.mean() * mask_1.sum() / batch_size

    # image2 wins: maximize P(r2 > r1)
    mask_2 = (winner == 1)
    if mask_2.any():
        l = (-F.logsigmoid(r2_type[mask_2] - r1_type[mask_2])) * weight[mask_2]
        loss = loss + l.mean() * mask_2.sum() / batch_size

    # tie: minimize |r1 - r2|
    mask_tie = (winner == 2)
    if mask_tie.any():
        l = torch.abs(r1_type[mask_tie] - r2_type[mask_tie]) * weight[mask_tie]
        loss = loss + l.mean() * mask_tie.sum() / batch_size

    # not_apply: push both low
    mask_na = (winner == 3)
    if mask_na.any():
        l = (F.relu(r1_type[mask_na]) + F.relu(r2_type[mask_na])) * weight[mask_na]
        loss = loss + l.mean() * mask_na.sum() / batch_size

    return loss


def load_data(test_split=0.2, random_seed=42):
    """Load all data sources with train/test split for pairwise data."""
    # Load pairwise data (v1.8)
    if PAIRWISE_CSV.exists():
        pairwise_df = pd.read_csv(PAIRWISE_CSV)
        print(f"Loaded {len(pairwise_df)} pairwise comparisons from v1.8")
    else:
        # Fallback to original CSV
        fallback_csv = CLASSIFIER2_ROOT / "Quantum Label Data - Pairwise_Comparison.csv"
        if fallback_csv.exists():
            pairwise_df = pd.read_csv(fallback_csv)
            print(f"Loaded {len(pairwise_df)} pairwise comparisons (fallback)")
        else:
            raise FileNotFoundError("No pairwise comparison CSV found")

    # Split pairwise data into train/test by unique image pairs
    # (so all comparisons for a pair are in same split)
    pairwise_df['pair_key'] = pairwise_df.apply(
        lambda r: tuple(sorted([r['Image1_Path'], r['Image2_Path']])), axis=1
    )
    unique_pairs = list(pairwise_df['pair_key'].unique())

    # Shuffle and split
    random.seed(random_seed)
    random.shuffle(unique_pairs)
    split_idx = int(len(unique_pairs) * (1 - test_split))
    train_pairs = set(unique_pairs[:split_idx])
    test_pairs = set(unique_pairs[split_idx:])

    train_pairwise_df = pairwise_df[pairwise_df['pair_key'].isin(train_pairs)].copy()
    test_pairwise_df = pairwise_df[pairwise_df['pair_key'].isin(test_pairs)].copy()

    print(f"  Train pairs: {len(train_pairs)} ({len(train_pairwise_df)} comparisons)")
    print(f"  Test pairs: {len(test_pairs)} ({len(test_pairwise_df)} comparisons)")

    # Count confidence values
    if 'Confidence' in train_pairwise_df.columns:
        conf_counts = train_pairwise_df['Confidence'].value_counts(dropna=False)
        print(f"  Confidence distribution: {dict(conf_counts)}")

    # Load bad images from v1.8 absolute scoring
    bad_images = []
    abs_csv = ABSOLUTE_CSV if ABSOLUTE_CSV.exists() else CLASSIFIER2_ROOT / "Quantum Label Data - Absolute_Scoring.csv"
    if abs_csv.exists():
        abs_df = pd.read_csv(abs_csv)
        bad_df = abs_df[abs_df['Reconstruction'].str.contains('Bad', na=False)]
        for _, row in bad_df.iterrows():
            path = DATA_ROOT / row['File_Path']
            if path.exists():
                bad_images.append(str(path))
    print(f"Loaded {len(bad_images)} bad images")

    # Load ideal images for ALL 5 types
    ideal_images = {t: [] for t in RECONSTRUCTION_TYPES}

    for rec_type, ideal_dir in IDEAL_DIRS.items():
        if ideal_dir.exists():
            # Support both .png and .bmp files
            for ext in ['*.png', '*.bmp']:
                for img in ideal_dir.glob(ext):
                    ideal_images[rec_type].append(str(img))

    print("Loaded ideal images:")
    for rec_type, imgs in ideal_images.items():
        print(f"  {rec_type}: {len(imgs)}")

    # Load trajectory images (for anchoring)
    trajectory_images = []
    for subdir in ['2022-02-04', '2022-02-06', '2022-04-11']:
        subdir_path = TRAJECTORY_DIR / subdir
        if subdir_path.exists():
            for img in subdir_path.glob('*.bmp'):
                trajectory_images.append(str(img))
    print(f"Loaded {len(trajectory_images)} trajectory images")

    return train_pairwise_df, test_pairwise_df, ideal_images, bad_images, trajectory_images


def train(epochs=30, batch_size=16, lr=1e-4, samples_per_epoch=500):
    """Train the model."""
    print("=" * 60)
    print("CLASSIFIER2 TRAINING (v2.0)")
    print("=" * 60)
    sys.stdout.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_pairwise_df, test_pairwise_df, ideal_images, bad_images, trajectory_images = load_data()

    # Create dataset (using only train pairwise data)
    dataset = UnifiedDataset(
        train_pairwise_df, ideal_images, bad_images, trajectory_images,
        transform=get_transform(training=True),
        samples_per_epoch=samples_per_epoch
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Store test data for later evaluation
    test_data = {
        'pairwise_df': test_pairwise_df,
        'ideal_images': ideal_images,
        'bad_images': bad_images
    }

    # Create model
    model = BradleyTerryModel(pretrained_encoder_path=str(PRETRAINED_ENCODER))
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    sys.stdout.flush()

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            loss = compute_loss(model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best model
            save_path = CLASSIFIER2_ROOT / "artifacts" / "best_model.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        sys.stdout.flush()

    print("-" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")

    # Save final model
    final_path = CLASSIFIER2_ROOT / "artifacts" / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")

    # Evaluate on held-out pairwise data
    print("\n" + "=" * 60)
    print("PAIRWISE HOLDOUT EVALUATION")
    print("=" * 60)
    evaluate_pairwise_holdout(model, test_data['pairwise_df'], device)

    return model


def evaluate_pairwise_holdout(model, test_pairwise_df, device):
    """Evaluate model on held-out pairwise comparisons."""
    model.eval()
    transform = get_transform(training=False)

    results_by_type = {rt: {'correct': 0, 'total': 0} for rt in RECONSTRUCTION_TYPES}
    results_by_winner = {'1': {'correct': 0, 'total': 0},
                         '2': {'correct': 0, 'total': 0},
                         'tie': {'correct': 0, 'total': 0},
                         'not_apply': {'correct': 0, 'total': 0}}

    with torch.no_grad():
        for idx in range(len(test_pairwise_df)):
            row = test_pairwise_df.iloc[idx]

            rt = row['Reconstruction_Type']
            winner = str(row['Winner'])

            if rt not in TYPE_TO_IDX or winner not in WINNER_MAP:
                continue

            type_idx = TYPE_TO_IDX[rt]

            # Load images
            path1 = DATA_ROOT / row['Image1_Path']
            path2 = DATA_ROOT / row['Image2_Path']

            if not path1.exists() or not path2.exists():
                continue

            img1 = Image.open(path1).convert('L')
            img2 = Image.open(path2).convert('L')
            img1 = transform(img1).unsqueeze(0).to(device)
            img2 = transform(img2).unsqueeze(0).to(device)

            # Get scores
            r1 = model(img1).squeeze()[type_idx].item()
            r2 = model(img2).squeeze()[type_idx].item()

            # Determine if prediction matches human judgment
            if winner == '1':
                correct = r1 > r2
            elif winner == '2':
                correct = r2 > r1
            elif winner == 'tie':
                # For ties, consider correct if scores are within margin
                correct = abs(r1 - r2) < 0.5
            elif winner == 'not_apply':
                # For not_apply, both should be low (below 0)
                correct = r1 < 0 and r2 < 0

            results_by_type[rt]['total'] += 1
            results_by_winner[winner]['total'] += 1
            if correct:
                results_by_type[rt]['correct'] += 1
                results_by_winner[winner]['correct'] += 1

    # Print results by reconstruction type
    print("\n--- Accuracy by Reconstruction Type ---")
    total_correct = 0
    total_all = 0
    for rt in RECONSTRUCTION_TYPES:
        stats = results_by_type[rt]
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {rt}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
            total_correct += stats['correct']
            total_all += stats['total']

    if total_all > 0:
        print(f"\n  Overall: {100*total_correct/total_all:.1f}% ({total_correct}/{total_all})")

    # Print results by winner type
    print("\n--- Accuracy by Winner Type ---")
    for winner_type in ['1', '2', 'tie', 'not_apply']:
        stats = results_by_winner[winner_type]
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            label = {'1': 'image1 wins', '2': 'image2 wins'}.get(winner_type, winner_type)
            print(f"  {label}: {acc:.1f}% ({stats['correct']}/{stats['total']})")


def test_model(model_path=None):
    """Test the model on Test and Val data."""
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = BradleyTerryModel()
    if model_path is None:
        model_path = CLASSIFIER2_ROOT / "artifacts" / "best_model.pth"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = get_transform(training=False)

    # Test on Test folder
    test_dir = DATA_ROOT / "Test"
    val_dir = DATA_ROOT / "Val"

    for folder_name, folder_path in [("Test", test_dir), ("Val", val_dir)]:
        if not folder_path.exists():
            continue

        print(f"\n--- {folder_name} Results ---")

        results = []
        for img_path in sorted(list(folder_path.glob('*.png')) + list(folder_path.glob('*.bmp'))):
            # Determine ground truth from filename
            name = img_path.stem
            if name.startswith('HTR'):
                gt_type = 'HTR'
                gt_idx = TYPE_TO_IDX['HTR']
            elif name.startswith('RT13'):
                gt_type = 'RT13'
                gt_idx = TYPE_TO_IDX['(√13 x √13)']
            else:
                gt_type = 'Unknown'
                gt_idx = -1

            # Load and predict
            img = Image.open(img_path).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                scores = model(img_tensor).squeeze().cpu().numpy()

            # Get predicted type
            pred_idx = np.argmax(scores)
            pred_type = RECONSTRUCTION_TYPES[pred_idx]

            # Check if correct (for HTR and RT13)
            correct = (gt_idx == pred_idx) if gt_idx >= 0 else None

            results.append({
                'file': img_path.name,
                'ground_truth': gt_type,
                'predicted': pred_type,
                'correct': correct,
                'scores': scores
            })

            # Print scores
            score_str = ' | '.join([f"{RECONSTRUCTION_TYPES[i][:6]}: {scores[i]:.2f}" for i in range(NUM_TYPES)])
            status = "✓" if correct else "✗" if correct is not None else "?"
            print(f"  {status} {img_path.name}: pred={pred_type}, scores=[{score_str}]")

        # Calculate accuracy
        valid_results = [r for r in results if r['correct'] is not None]
        if valid_results:
            accuracy = sum(1 for r in valid_results if r['correct']) / len(valid_results)
            print(f"\n  Accuracy: {accuracy:.1%} ({sum(1 for r in valid_results if r['correct'])}/{len(valid_results)})")

    return results


if __name__ == '__main__':
    # Train
    model = train(epochs=30, batch_size=16, lr=1e-4, samples_per_epoch=500)

    # Test
    test_model()
