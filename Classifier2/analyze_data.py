#!/usr/bin/env python3
"""
Analyze all available data for Classifier2 training.

Data sources:
1. Pairwise comparison data (from labeling software)
2. Absolute scoring data - Bad images
3. Ideal images (STO_ideal_HTR, STO_ideal_RT13)
4. Trajectory images (unlabeled pool)
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CLASSIFIER2_ROOT = Path(__file__).parent

# CSV files
PAIRWISE_CSV = CLASSIFIER2_ROOT / "Quantum Label Data - Pairwise_Comparison.csv"
ABSOLUTE_CSV = CLASSIFIER2_ROOT / "Quantum Label Data - Absolute_Scoring.csv"


def analyze_pairwise_data():
    """Analyze pairwise comparison data."""
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISON DATA")
    print("=" * 70)

    if not PAIRWISE_CSV.exists():
        print(f"File not found: {PAIRWISE_CSV}")
        return None

    df = pd.read_csv(PAIRWISE_CSV)
    print(f"\nTotal pairwise comparisons: {len(df)}")

    # Unique image pairs (each pair has 5 comparisons, one per reconstruction type)
    df['pair_key'] = df.apply(lambda r: tuple(sorted([r['Image1_Path'], r['Image2_Path']])), axis=1)
    unique_pairs = df['pair_key'].nunique()
    print(f"Unique image pairs compared: {unique_pairs}")

    # Unique images
    all_images = set(df['Image1_Path'].tolist() + df['Image2_Path'].tolist())
    print(f"Unique images in comparisons: {len(all_images)}")

    # Per reconstruction type
    print("\n--- Comparisons per Reconstruction Type ---")
    for rt in df['Reconstruction_Type'].unique():
        count = len(df[df['Reconstruction_Type'] == rt])
        print(f"  {rt}: {count}")

    # Winner distribution
    print("\n--- Winner Distribution ---")
    winner_counts = df['Winner'].value_counts()
    for winner, count in winner_counts.items():
        pct = 100 * count / len(df)
        print(f"  {winner}: {count} ({pct:.1f}%)")

    # Per reconstruction type winner distribution
    print("\n--- Winner Distribution per Reconstruction Type ---")
    for rt in sorted(df['Reconstruction_Type'].unique()):
        rt_df = df[df['Reconstruction_Type'] == rt]
        print(f"\n  {rt}:")
        for winner in ['1', '2', 'tie', 'not_apply']:
            count = len(rt_df[rt_df['Winner'] == winner])
            pct = 100 * count / len(rt_df) if len(rt_df) > 0 else 0
            label = {'1': 'image1', '2': 'image2'}.get(winner, winner)
            print(f"    {label}: {count} ({pct:.1f}%)")

    # Labelers
    print("\n--- Labelers ---")
    for labeler in df['Labeler_Name'].unique():
        if labeler:
            count = len(df[df['Labeler_Name'] == labeler])
            print(f"  {labeler}: {count} comparisons")

    return df


def analyze_absolute_data():
    """Analyze absolute scoring data (Bad images)."""
    print("\n" + "=" * 70)
    print("ABSOLUTE SCORING DATA (Bad Images)")
    print("=" * 70)

    if not ABSOLUTE_CSV.exists():
        print(f"File not found: {ABSOLUTE_CSV}")
        return None

    df = pd.read_csv(ABSOLUTE_CSV)
    print(f"\nTotal entries: {len(df)}")

    # Filter to Bad images
    bad_df = df[df['Reconstruction'].str.contains('Bad', na=False)]
    print(f"Bad images: {len(bad_df)}")

    # List bad images
    print("\n--- Bad Images ---")
    for _, row in bad_df.iterrows():
        print(f"  {row['File_Name']}")

    return df


def analyze_ideal_data():
    """Analyze ideal/reference images."""
    print("\n" + "=" * 70)
    print("IDEAL/REFERENCE IMAGES")
    print("=" * 70)

    ideal_dirs = {
        'HTR': DATA_ROOT / 'STO_ideal_HTR',
        'RT13 (√13 x √13)': DATA_ROOT / 'STO_ideal_RT13',
    }

    total_ideal = 0
    for name, dir_path in ideal_dirs.items():
        if dir_path.exists():
            images = list(dir_path.glob('*.png')) + list(dir_path.glob('*.bmp'))
            print(f"\n  {name}: {len(images)} images")
            total_ideal += len(images)
        else:
            print(f"\n  {name}: Directory not found ({dir_path})")

    print(f"\nTotal ideal images: {total_ideal}")
    return ideal_dirs


def analyze_trajectory_data():
    """Analyze trajectory images (unlabeled pool)."""
    print("\n" + "=" * 70)
    print("TRAJECTORY IMAGES (Unlabeled Pool)")
    print("=" * 70)

    traj_dir = DATA_ROOT / 'Trajectories'

    if not traj_dir.exists():
        print(f"Directory not found: {traj_dir}")
        return

    total = 0
    for subdir in sorted(traj_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            images = list(subdir.glob('*.bmp')) + list(subdir.glob('*.png'))
            print(f"\n  {subdir.name}: {len(images)} images")
            total += len(images)

    print(f"\nTotal trajectory images: {total}")


def compute_coverage():
    """Compute what percentage of trajectory images have been labeled."""
    print("\n" + "=" * 70)
    print("LABELING COVERAGE")
    print("=" * 70)

    # Get all trajectory images
    traj_dir = DATA_ROOT / 'Trajectories'
    all_traj_images = set()
    for subdir in ['2022-02-04', '2022-02-06', '2022-04-11']:
        subdir_path = traj_dir / subdir
        if subdir_path.exists():
            for img in subdir_path.glob('*.bmp'):
                # Store relative path matching CSV format
                rel_path = f"Trajectories/{subdir}/{img.name}"
                all_traj_images.add(rel_path)

    print(f"\nTotal trajectory images (2022 folders): {len(all_traj_images)}")

    # Get labeled images from pairwise data
    pairwise_images = set()
    if PAIRWISE_CSV.exists():
        df = pd.read_csv(PAIRWISE_CSV)
        pairwise_images = set(df['Image1_Path'].tolist() + df['Image2_Path'].tolist())

    # Get bad images
    bad_images = set()
    if ABSOLUTE_CSV.exists():
        df = pd.read_csv(ABSOLUTE_CSV)
        bad_df = df[df['Reconstruction'].str.contains('Bad', na=False)]
        bad_images = set(bad_df['File_Path'].tolist())

    labeled = pairwise_images | bad_images
    coverage = 100 * len(labeled & all_traj_images) / len(all_traj_images) if all_traj_images else 0

    print(f"Images with pairwise labels: {len(pairwise_images)}")
    print(f"Images marked as Bad: {len(bad_images)}")
    print(f"Total unique labeled images: {len(labeled)}")
    print(f"Coverage of 2022 trajectories: {coverage:.1f}%")

    # Unlabeled images
    unlabeled = all_traj_images - labeled
    print(f"Unlabeled images: {len(unlabeled)}")


def print_data_summary():
    """Print comprehensive data summary."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY FOR CLASSIFIER2")
    print("=" * 70)

    summary = """
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PAIRWISE COMPARISON DATA (Classifier2)                          │
│     - 234 total comparisons                                          │
│     - ~47 unique image pairs                                         │
│     - 5 reconstruction types compared per pair                       │
│     - Winner types: image1, image2, tie, not_apply                  │
│                                                                      │
│  2. BAD IMAGE DATA (Absolute Scoring)                               │
│     - 13 images marked as "Bad"                                      │
│     - Useful as 6th class for quality filtering                     │
│                                                                      │
│  3. IDEAL IMAGES (Classifier1 reference)                            │
│     - STO_ideal_HTR: 19 images                                       │
│     - STO_ideal_RT13: 21 images                                      │
│     - High-quality reference examples                                │
│                                                                      │
│  4. TRAJECTORY IMAGES (Unlabeled pool)                              │
│     - 2022-02-04: 388 images                                         │
│     - 2022-02-06: 287 images                                         │
│     - 2022-04-11: 449 images                                         │
│     - Total: ~1,124 images                                           │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                     RECONSTRUCTION TYPES                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. (1 x 1)          - Basic reconstruction                         │
│  2. Twinned(2 x 1)   - Twinned pattern                              │
│  3. c(6 x 2)         - Complex pattern                              │
│  4. (√13 x √13)      - RT13 pattern (has ideal images)             │
│  5. HTR              - High temperature reconstruction               │
│  6. Bad              - Low quality / unusable                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
"""
    print(summary)


def main():
    print("=" * 70)
    print("CLASSIFIER2 DATA ANALYSIS")
    print("=" * 70)

    # Analyze each data source
    pairwise_df = analyze_pairwise_data()
    absolute_df = analyze_absolute_data()
    analyze_ideal_data()
    analyze_trajectory_data()
    compute_coverage()
    print_data_summary()


if __name__ == '__main__':
    main()
