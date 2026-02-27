#!/usr/bin/env python3
"""
Learning Curve Analysis for Classifier2

Trains the model with increasing amounts of data (10%, 20%, ..., 100%)
to determine if we're hitting a plateau or need more data.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# Import from train_unified
from train_unified import (
    PROJECT_ROOT, DATA_ROOT, CLASSIFIER2_ROOT, CLASSIFIER1_ROOT,
    PAIRWISE_CSV, ABSOLUTE_CSV, IDEAL_HTR_DIR, IDEAL_RT13_DIR, TRAJECTORY_DIR,
    PRETRAINED_ENCODER, RECONSTRUCTION_TYPES, NUM_TYPES, TYPE_TO_IDX, WINNER_MAP,
    get_transform, UnifiedDataset, BradleyTerryModel, compute_loss
)


def load_data_with_fraction(train_fraction=1.0, test_split=0.2, random_seed=42):
    """
    Load data with a fraction of training pairs.

    Args:
        train_fraction: Fraction of training pairs to use (0.0 to 1.0)
        test_split: Fraction of data to hold out for testing
        random_seed: Random seed for reproducibility
    """
    # Load pairwise data
    pairwise_df = pd.read_csv(PAIRWISE_CSV)

    # Split by unique pairs
    pairwise_df['pair_key'] = pairwise_df.apply(
        lambda r: tuple(sorted([r['Image1_Path'], r['Image2_Path']])), axis=1
    )
    unique_pairs = list(pairwise_df['pair_key'].unique())

    # Shuffle with fixed seed for consistent test set
    random.seed(random_seed)
    random.shuffle(unique_pairs)

    # Split into train/test
    split_idx = int(len(unique_pairs) * (1 - test_split))
    all_train_pairs = unique_pairs[:split_idx]
    test_pairs = set(unique_pairs[split_idx:])

    # Take fraction of training pairs
    num_train_pairs = max(1, int(len(all_train_pairs) * train_fraction))
    train_pairs = set(all_train_pairs[:num_train_pairs])

    train_pairwise_df = pairwise_df[pairwise_df['pair_key'].isin(train_pairs)].copy()
    test_pairwise_df = pairwise_df[pairwise_df['pair_key'].isin(test_pairs)].copy()

    # Load supporting data
    bad_images = []
    if ABSOLUTE_CSV.exists():
        abs_df = pd.read_csv(ABSOLUTE_CSV)
        bad_df = abs_df[abs_df['Reconstruction'].str.contains('Bad', na=False)]
        for _, row in bad_df.iterrows():
            path = DATA_ROOT / row['File_Path']
            if path.exists():
                bad_images.append(str(path))

    ideal_images = {'HTR': [], 'RT13': []}
    if IDEAL_HTR_DIR.exists():
        for img in IDEAL_HTR_DIR.glob('*.png'):
            ideal_images['HTR'].append(str(img))
    if IDEAL_RT13_DIR.exists():
        for img in IDEAL_RT13_DIR.glob('*.png'):
            ideal_images['RT13'].append(str(img))

    trajectory_images = []
    for subdir in ['2022-02-04', '2022-02-06', '2022-04-11']:
        subdir_path = TRAJECTORY_DIR / subdir
        if subdir_path.exists():
            for img in subdir_path.glob('*.bmp'):
                trajectory_images.append(str(img))

    return {
        'train_df': train_pairwise_df,
        'test_df': test_pairwise_df,
        'ideal_images': ideal_images,
        'bad_images': bad_images,
        'trajectory_images': trajectory_images,
        'num_train_pairs': len(train_pairs),
        'num_train_comparisons': len(train_pairwise_df),
        'num_test_pairs': len(test_pairs),
        'num_test_comparisons': len(test_pairwise_df)
    }


def evaluate_model(model, test_df, device):
    """Evaluate model and return accuracy."""
    model.eval()
    transform = get_transform(training=False)

    total_correct = 0
    total = 0

    with torch.no_grad():
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]

            rt = row['Reconstruction_Type']
            winner = str(row['Winner'])

            if rt not in TYPE_TO_IDX or winner not in WINNER_MAP:
                continue

            type_idx = TYPE_TO_IDX[rt]

            path1 = DATA_ROOT / row['Image1_Path']
            path2 = DATA_ROOT / row['Image2_Path']

            if not path1.exists() or not path2.exists():
                continue

            img1 = Image.open(path1).convert('L')
            img2 = Image.open(path2).convert('L')
            img1 = transform(img1).unsqueeze(0).to(device)
            img2 = transform(img2).unsqueeze(0).to(device)

            r1 = model(img1).squeeze()[type_idx].item()
            r2 = model(img2).squeeze()[type_idx].item()

            if winner == '1':
                correct = r1 > r2
            elif winner == '2':
                correct = r2 > r1
            elif winner == 'tie':
                correct = abs(r1 - r2) < 0.5
            elif winner == 'not_apply':
                correct = r1 < 0 and r2 < 0
            else:
                continue

            total += 1
            if correct:
                total_correct += 1

    return total_correct / total if total > 0 else 0


def train_with_fraction(train_fraction, epochs=30, batch_size=16, lr=1e-4,
                        samples_per_epoch=300, random_seed=42, verbose=False):
    """Train model with a fraction of data and return metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data = load_data_with_fraction(train_fraction, random_seed=random_seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training with {train_fraction*100:.0f}% of data")
        print(f"Train pairs: {data['num_train_pairs']}, comparisons: {data['num_train_comparisons']}")
        print(f"{'='*60}")

    # Create dataset
    dataset = UnifiedDataset(
        data['train_df'], data['ideal_images'], data['bad_images'],
        data['trajectory_images'],
        transform=get_transform(training=True),
        samples_per_epoch=samples_per_epoch
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    model = BradleyTerryModel(pretrained_encoder_path=str(PRETRAINED_ENCODER))
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
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

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    # Evaluate
    accuracy = evaluate_model(model, data['test_df'], device)

    return {
        'train_fraction': train_fraction,
        'num_train_pairs': data['num_train_pairs'],
        'num_train_comparisons': data['num_train_comparisons'],
        'final_loss': best_loss,
        'holdout_accuracy': accuracy
    }


def run_learning_curve(fractions=None, epochs=30, runs_per_fraction=3):
    """
    Run learning curve experiment.

    Args:
        fractions: List of fractions to test (default: 0.1, 0.2, ..., 1.0)
        epochs: Training epochs per run
        runs_per_fraction: Number of runs to average (for stability)
    """
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("=" * 60)
    print("LEARNING CURVE ANALYSIS")
    print("=" * 60)
    print(f"Fractions to test: {fractions}")
    print(f"Epochs per run: {epochs}")
    print(f"Runs per fraction: {runs_per_fraction}")
    print("=" * 60)
    sys.stdout.flush()

    results = []

    for frac in fractions:
        print(f"\n>>> Testing {frac*100:.0f}% of training data...")
        sys.stdout.flush()

        frac_results = []
        for run in range(runs_per_fraction):
            # Use different seed for each run
            seed = 42 + run
            result = train_with_fraction(
                frac, epochs=epochs, random_seed=seed, verbose=(run == 0)
            )
            frac_results.append(result)
            print(f"    Run {run+1}: {result['holdout_accuracy']*100:.1f}% accuracy")
            sys.stdout.flush()

        # Average results
        avg_accuracy = np.mean([r['holdout_accuracy'] for r in frac_results])
        std_accuracy = np.std([r['holdout_accuracy'] for r in frac_results])

        results.append({
            'fraction': frac,
            'num_pairs': frac_results[0]['num_train_pairs'],
            'num_comparisons': frac_results[0]['num_train_comparisons'],
            'mean_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'all_runs': [r['holdout_accuracy'] for r in frac_results]
        })

        print(f"    Average: {avg_accuracy*100:.1f}% ± {std_accuracy*100:.1f}%")

    # Print summary table
    print("\n" + "=" * 60)
    print("LEARNING CURVE RESULTS")
    print("=" * 60)
    print(f"{'Fraction':<10} {'Pairs':<8} {'Comparisons':<12} {'Accuracy':<15}")
    print("-" * 60)

    for r in results:
        acc_str = f"{r['mean_accuracy']*100:.1f}% ± {r['std_accuracy']*100:.1f}%"
        print(f"{r['fraction']*100:>6.0f}%    {r['num_pairs']:<8} {r['num_comparisons']:<12} {acc_str:<15}")

    # Save results
    output_path = CLASSIFIER2_ROOT / "artifacts" / "learning_curve_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    accuracies = [r['mean_accuracy'] for r in results]

    # Check if plateauing
    if len(accuracies) >= 3:
        last_improvement = accuracies[-1] - accuracies[-2]
        avg_early_improvement = np.mean([accuracies[i+1] - accuracies[i]
                                         for i in range(min(3, len(accuracies)-1))])

        if last_improvement < avg_early_improvement * 0.3:
            print("PLATEAU DETECTED: Recent improvements are small compared to early gains.")
            print("This suggests the model architecture may be the limiting factor.")
        else:
            print("NO PLATEAU: Accuracy is still improving with more data.")
            print("More labeled data would likely improve performance.")

    return results


if __name__ == '__main__':
    # Run with fewer epochs for speed, but multiple runs for stability
    results = run_learning_curve(
        fractions=[0.2, 0.4, 0.6, 0.8, 1.0],  # 5 points for faster run
        epochs=30,
        runs_per_fraction=2  # Average over 2 runs
    )
