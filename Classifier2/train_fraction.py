#!/usr/bin/env python3
"""
Train with a specific fraction of data.
Usage: python train_fraction.py <fraction> [--seed <seed>]
"""

import sys
import argparse
from learning_curve import train_with_fraction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fraction', type=float, help='Fraction of data (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    args = parser.parse_args()

    result = train_with_fraction(
        args.fraction,
        epochs=args.epochs,
        random_seed=args.seed,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"RESULT: {args.fraction*100:.0f}% data")
    print(f"{'='*60}")
    print(f"Train pairs: {result['num_train_pairs']}")
    print(f"Train comparisons: {result['num_train_comparisons']}")
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Holdout accuracy: {result['holdout_accuracy']*100:.1f}%")
