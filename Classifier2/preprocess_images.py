#!/usr/bin/env python3
"""
Preprocess RHEED images for Classifier2 training.

Two preprocessing steps:
1. Apply 350% brightness to all BMP images (like labeling software)
2. Convert black images in 2025 folders to green (to match 2022 trajectory format)
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAJECTORY_DIR = DATA_ROOT / "Trajectories"

# Backup directory for original images
BACKUP_DIR = DATA_ROOT / "Trajectories_backup"


def convert_grayscale_to_green(img):
    """
    Convert grayscale image to green phosphor format (like 2022 images).

    The 2022 images use a palette with green phosphor colors:
    - Index 0-127: Black to bright green (0, 0, 0) -> (0, 255, 0)
    - Index 128-255: Bright green to white (0, 255, 0) -> (255, 255, 255)

    The 2025 images are pure grayscale (L mode).
    This function converts grayscale to green phosphor format.
    """
    # Ensure we have grayscale values
    if img.mode == 'P':
        # Already has a palette, convert to RGB first
        img = img.convert('RGB')
        arr = np.array(img)
        # If it's already green-dominant, return as-is
        if arr[:, :, 1].mean() > arr[:, :, 0].mean() * 1.5:
            return img
        intensity = arr[:, :, 1]  # Use green channel
    elif img.mode == 'L':
        intensity = np.array(img)
    elif img.mode in ['RGB', 'RGBA']:
        arr = np.array(img)
        # Check if already green format
        if arr[:, :, 1].mean() > arr[:, :, 0].mean() * 1.5:
            return img.convert('RGB')
        # Use luminance
        intensity = np.mean(arr[:, :, :3], axis=2).astype(np.uint8)
    else:
        img = img.convert('L')
        intensity = np.array(img)

    # Create green phosphor image
    # Map intensity to green palette:
    # - Low values (0-127): pure green gradient
    # - High values (128-255): green to white gradient
    h, w = intensity.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Scale intensity to match phosphor response
    # Multiply by 2 to boost the green channel (like palette index * 2)
    green_value = np.clip(intensity * 2, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = green_value  # Green channel

    # For very bright areas (intensity > 127), add some R and B
    bright_mask = intensity > 127
    overflow = np.clip((intensity.astype(np.int16) - 127) * 2, 0, 255).astype(np.uint8)
    rgb[:, :, 0][bright_mask] = overflow[bright_mask]
    rgb[:, :, 2][bright_mask] = overflow[bright_mask]

    return Image.fromarray(rgb, mode='RGB')


def apply_brightness(img, factor=3.5):
    """
    Apply brightness enhancement (350% = factor of 3.5).
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def needs_green_conversion(img):
    """
    Check if image needs to be converted to green phosphor format.
    Returns True for grayscale images (mode L) or images without green dominance.
    """
    if img.mode == 'L':
        return True  # Grayscale needs conversion

    if img.mode == 'P':
        # Check palette - convert to RGB to analyze
        rgb = img.convert('RGB')
        arr = np.array(rgb)
    elif img.mode in ['RGB', 'RGBA']:
        arr = np.array(img)
    else:
        return True  # Unknown mode, convert

    # Check if green is already dominant
    green = arr[:, :, 1].mean()
    red = arr[:, :, 0].mean()
    blue = arr[:, :, 2].mean()

    # If green is dominant (like 2022 images), no conversion needed
    if green > max(red, blue) * 1.5 and green > 1:
        return False

    return True


def preprocess_all_images(brightness_factor=1.5, backup=True):
    """
    Preprocess BMP images in the data folder.

    IMPORTANT: Only applies brightness to TRAJECTORY images.
    Ideal images and Test/Val images are left unchanged to preserve their calibration.

    Args:
        brightness_factor: Brightness multiplier (1.5 = 150%, default)
        backup: Whether to create backup of original images
    """
    print("=" * 60)
    print("RHEED IMAGE PREPROCESSING")
    print("=" * 60)
    print(f"Brightness factor: {brightness_factor}x ({brightness_factor*100:.0f}%)")
    print(f"Data root: {DATA_ROOT}")
    print(f"NOTE: Only trajectory images will be brightness-adjusted")
    sys.stdout.flush()

    # Find trajectory BMP files ONLY (not ideal/test/val)
    trajectory_bmps = []

    # Trajectory images only
    for folder in TRAJECTORY_DIR.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            # Handle subfolders (like 2025-10-04/A)
            for item in folder.rglob('*.bmp'):
                trajectory_bmps.append(item)

    all_bmps = trajectory_bmps
    print(f"\nFound {len(all_bmps)} trajectory images to process")
    print(f"(Ideal and Test/Val images will NOT be modified)")
    sys.stdout.flush()

    # Identify 2025 folder images (need color conversion)
    folders_2025 = ['2025-10-04', '2025-10-05']

    # Process images
    converted_count = 0
    brightness_count = 0
    error_count = 0

    for img_path in tqdm(all_bmps, desc="Processing images"):
        try:
            # Load image
            img = Image.open(img_path)

            # Check if this is a 2025 image that needs color conversion
            is_2025 = any(folder in str(img_path) for folder in folders_2025)

            if is_2025 and needs_green_conversion(img):
                # Convert grayscale to green phosphor format
                img = convert_grayscale_to_green(img)
                converted_count += 1
            else:
                # For non-2025 images, just convert to RGB if needed
                if img.mode == 'P':
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')
                elif img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')

            # Apply brightness to all images
            img = apply_brightness(img, brightness_factor)
            brightness_count += 1

            # Save back (overwrite original)
            img.save(img_path)

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            error_count += 1
            sys.stdout.flush()

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Images processed: {len(all_bmps)}")
    print(f"Brightness adjusted: {brightness_count}")
    print(f"Black->Green converted: {converted_count}")
    print(f"Errors: {error_count}")
    sys.stdout.flush()


def preview_preprocessing(sample_size=3):
    """
    Preview preprocessing on a few sample images without saving.
    """
    print("=" * 60)
    print("PREPROCESSING PREVIEW")
    print("=" * 60)

    # Sample from 2022 and 2025 folders
    samples = {
        '2022-02-04': [],
        '2025-10-04': [],
    }

    for folder_name in samples.keys():
        folder_path = TRAJECTORY_DIR / folder_name
        if folder_path.exists():
            # Find BMPs (might be in subfolders)
            bmps = list(folder_path.rglob('*.bmp'))[:sample_size]
            samples[folder_name] = bmps

    for folder_name, files in samples.items():
        print(f"\n--- {folder_name} ---")
        for f in files:
            try:
                img = Image.open(f)
                arr = np.array(img)

                print(f"\n  {f.name}")
                print(f"    Size: {img.size}")
                print(f"    Mode: {img.mode}")
                print(f"    Shape: {arr.shape}")

                if len(arr.shape) == 3 and arr.shape[2] >= 3:
                    print(f"    R mean: {arr[:,:,0].mean():.1f}")
                    print(f"    G mean: {arr[:,:,1].mean():.1f}")
                    print(f"    B mean: {arr[:,:,2].mean():.1f}")
                else:
                    print(f"    Mean intensity: {arr.mean():.1f}")

            except Exception as e:
                print(f"    Error: {e}")

    sys.stdout.flush()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess RHEED images')
    parser.add_argument('--preview', action='store_true', help='Preview without modifying')
    parser.add_argument('--brightness', type=float, default=1.5, help='Brightness factor (default: 1.5 = 150%%)')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup')

    args = parser.parse_args()

    if args.preview:
        preview_preprocessing()
    else:
        preprocess_all_images(
            brightness_factor=args.brightness,
            backup=not args.no_backup
        )
