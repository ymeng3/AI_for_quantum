#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script for Classifier2 (v2.1)

Five evaluation modes:
1. Pairwise comparison: Input two images, output which better represents a reconstruction type
2. Win-rate classification: Compare test image against ideal references using proper win-rate
3. Bad detection: Detect low-quality/bad images
4. Argmax classification: Direct argmax on reward scores (not principled, for comparison only)
5. Ideal split evaluation: Split ideal images into reference/test for proper 4-class evaluation

Current data (January 2025):
- Ideal HTR: 21 images
- Ideal RT13: 21 images
- Ideal 1x1: 27 images
- Ideal c6x2: 24 images
- Ideal Twinned: 6 images (excluded from ideal-split due to small sample size)
- Bad images: 18

Latest results (4-class ideal split evaluation):
- Overall accuracy: 82.4% (14/17)
- HTR: 100% (4/4)
- c6x2: 100% (4/4)
- 1x1: 80% (4/5)
- RT13: 50% (2/4)

Comparison to baseline (SimCLR encoder without pairwise training): 41.2%
Pairwise training approximately doubles single-image classification accuracy!
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F

from train_unified import (
    DATA_ROOT, CLASSIFIER2_ROOT, PAIRWISE_CSV, ABSOLUTE_CSV,
    IDEAL_DIRS, IDEAL_HTR_DIR, IDEAL_RT13_DIR,
    RECONSTRUCTION_TYPES, NUM_TYPES, TYPE_TO_IDX, WINNER_MAP,
    get_transform, BradleyTerryModel
)


def load_model(model_path=None):
    """Load trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BradleyTerryModel()
    if model_path is None:
        model_path = CLASSIFIER2_ROOT / "artifacts" / "best_model.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device


def load_all_ideal_scores(model, device):
    """
    Load ALL ideal image reward scores (full 5-dim vectors).
    This allows cross-type comparison for win-rate classification.
    """
    transform = get_transform(training=False)

    all_ideal_scores = {t: [] for t in RECONSTRUCTION_TYPES}

    for rec_type, ideal_dir in IDEAL_DIRS.items():
        if ideal_dir.exists():
            for ext in ['*.png', '*.bmp']:
                for img_path in ideal_dir.glob(ext):
                    img = Image.open(img_path).convert('L')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        scores = model(img_tensor).squeeze().cpu().numpy()
                    all_ideal_scores[rec_type].append(scores)

    return all_ideal_scores


def load_bad_image_scores(model, device):
    """Load scores for bad reference images."""
    transform = get_transform(training=False)
    bad_scores = []

    abs_csv = ABSOLUTE_CSV if ABSOLUTE_CSV.exists() else CLASSIFIER2_ROOT / "Quantum Label Data - Absolute_Scoring.csv"
    if abs_csv.exists():
        abs_df = pd.read_csv(abs_csv)
        bad_df = abs_df[abs_df['Reconstruction'].str.contains('Bad', na=False)]
        for _, row in bad_df.iterrows():
            path = DATA_ROOT / row['File_Path']
            if path.exists():
                img = Image.open(path).convert('L')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    scores = model(img_tensor).squeeze().cpu().numpy()
                bad_scores.append(scores)

    return bad_scores


def compute_win_rate(test_score, ref_scores, dim_idx):
    """
    Compute win-rate: average probability of beating each reference.

    Win-rate = (1/N) Σᵢ σ(test_score[dim] - ref_i[dim])

    This is the CORRECT sigmoid-then-mean approach.
    """
    if not ref_scores:
        return None

    probs = []
    for ref_score in ref_scores:
        diff = test_score[dim_idx] - ref_score[dim_idx]
        prob = 1 / (1 + np.exp(-diff))  # sigmoid
        probs.append(prob)

    return np.mean(probs)


def classify_winrate(model, image_path, device, all_ideal_scores, bad_scores=None, bad_threshold=0.6):
    """
    Classify a single image using WIN-RATE method (principled approach).

    For each candidate type T:
    1. Compute win-rate against OTHER-type references on dimension T
    2. Predicted class = type with highest cross-type win-rate

    Also detects "Bad" images by comparing against bad references.

    Args:
        model: Trained model
        image_path: Path to image
        device: torch device
        all_ideal_scores: Dict from load_all_ideal_scores()
        bad_scores: List of score vectors for bad reference images
        bad_threshold: Threshold for classifying as "Bad"

    Returns:
        dict with classification results
    """
    transform = get_transform(training=False)

    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        test_scores = model(img_tensor).squeeze().cpu().numpy()

    # Step 1: Check if image is "Bad"
    is_bad = False
    bad_confidence = 0.0
    if bad_scores:
        # For each dimension, compute P(bad_ref > test)
        # If bad refs typically beat test, then test is probably bad
        bad_win_rates = []
        for dim_idx in range(NUM_TYPES):
            bad_probs = []
            for bad_score in bad_scores:
                diff = bad_score[dim_idx] - test_scores[dim_idx]
                prob = 1 / (1 + np.exp(-diff))
                bad_probs.append(prob)
            bad_win_rates.append(np.mean(bad_probs))

        bad_confidence = np.mean(bad_win_rates)
        is_bad = bad_confidence > bad_threshold

    if is_bad:
        return {
            'predicted_class': 'Bad',
            'predicted_idx': -1,
            'raw_scores': {RECONSTRUCTION_TYPES[i]: float(test_scores[i]) for i in range(NUM_TYPES)},
            'classification_scores': {'Bad': bad_confidence},
            'quality': None,
            'method': 'win-rate with bad detection',
            'is_bad': True,
            'bad_confidence': bad_confidence
        }

    # Step 2: Cross-type win-rate classification
    classification_scores = {}
    details = {}

    for rec_type in RECONSTRUCTION_TYPES:
        type_idx = TYPE_TO_IDX[rec_type]

        # Get OTHER-type references (opponents)
        opponents = []
        for other_type, scores_list in all_ideal_scores.items():
            if other_type != rec_type:
                opponents.extend(scores_list)

        if opponents:
            win_rate = compute_win_rate(test_scores, opponents, type_idx)
            classification_scores[rec_type] = win_rate
            details[f'win_rate_{rec_type}'] = win_rate

    # Predict class with highest cross-type win-rate
    if classification_scores:
        pred_class = max(classification_scores, key=classification_scores.get)
        pred_idx = TYPE_TO_IDX[pred_class]
    else:
        pred_class = None
        pred_idx = -1

    # Step 3: Quality score (same-type win-rate)
    quality = None
    if pred_class and all_ideal_scores.get(pred_class):
        quality = compute_win_rate(
            test_scores,
            all_ideal_scores[pred_class],
            TYPE_TO_IDX[pred_class]
        )

    return {
        'predicted_class': pred_class,
        'predicted_idx': pred_idx,
        'raw_scores': {RECONSTRUCTION_TYPES[i]: float(test_scores[i]) for i in range(NUM_TYPES)},
        'classification_scores': classification_scores,
        'quality': quality,
        'details': details,
        'method': 'win-rate (sigmoid-then-mean)',
        'is_bad': False,
        'bad_confidence': bad_confidence
    }


def classify_argmax(model, image_path, device):
    """
    Classify using simple argmax (NOT principled - for comparison only).

    WARNING: This is not a principled approach because reward scores
    are not calibrated across reconstruction types.
    """
    transform = get_transform(training=False)

    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(img_tensor).squeeze().cpu().numpy()

    pred_idx = np.argmax(scores)
    pred_class = RECONSTRUCTION_TYPES[pred_idx]

    return {
        'predicted_class': pred_class,
        'predicted_idx': pred_idx,
        'scores': {RECONSTRUCTION_TYPES[i]: float(scores[i]) for i in range(NUM_TYPES)},
        'confidence': float(F.softmax(torch.tensor(scores), dim=0)[pred_idx]),
        'method': 'argmax (not principled)'
    }


def compare_pair(model, image1_path, image2_path, reconstruction_type, device):
    """
    Compare two images for a specific reconstruction type.

    Args:
        model: Trained model
        image1_path: Path to first image
        image2_path: Path to second image
        reconstruction_type: Which reconstruction type to compare on
        device: torch device

    Returns:
        dict with comparison results
    """
    transform = get_transform(training=False)

    img1 = Image.open(image1_path).convert('L')
    img2 = Image.open(image2_path).convert('L')

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    type_idx = TYPE_TO_IDX[reconstruction_type]

    with torch.no_grad():
        r1 = model(img1).squeeze()[type_idx].item()
        r2 = model(img2).squeeze()[type_idx].item()

    prob_1_wins = 1 / (1 + np.exp(-(r1 - r2)))

    return {
        'reconstruction_type': reconstruction_type,
        'winner': 1 if r1 > r2 else 2,
        'score1': float(r1),
        'score2': float(r2),
        'prob_1_wins': float(prob_1_wins),
        'confidence': float(max(prob_1_wins, 1 - prob_1_wins))
    }


def evaluate_single_image_mode(model, device, test_dir=None):
    """
    Evaluate model in single-image classification mode.
    Uses Test and Val folders with labeled images.
    Compares win-rate method vs argmax method.

    Args:
        model: Trained model
        device: torch device
        test_dir: Optional custom test directory
    """
    print("\n" + "=" * 60)
    print("SINGLE-IMAGE CLASSIFICATION EVALUATION (v2.0)")
    print("=" * 60)

    # Load ideal reference scores for win-rate classification
    all_ideal_scores = load_all_ideal_scores(model, device)
    bad_scores = load_bad_image_scores(model, device)

    print(f"\nLoaded ideal references:")
    for rt, scores_list in all_ideal_scores.items():
        if scores_list:
            print(f"  {rt}: {len(scores_list)} images")
    print(f"  Bad references: {len(bad_scores)} images")

    if test_dir is None:
        test_dirs = [
            ("Test", DATA_ROOT / "Test"),
            ("Val", DATA_ROOT / "Val")
        ]
    else:
        test_dirs = [("Custom", Path(test_dir))]

    all_results = {'winrate': [], 'argmax': []}

    for folder_name, folder_path in test_dirs:
        if not folder_path.exists():
            print(f"\n{folder_name} folder not found: {folder_path}")
            continue

        print(f"\n{'='*60}")
        print(f"{folder_name} Results")
        print(f"{'='*60}")

        winrate_results = []
        argmax_results = []

        for img_path in sorted(list(folder_path.glob('*.png')) + list(folder_path.glob('*.bmp'))):
            # Determine ground truth from filename
            name = img_path.stem.upper()

            if 'HTR' in name:
                gt_type = 'HTR'
                gt_idx = TYPE_TO_IDX['HTR']
            elif 'RT13' in name or '√13' in name or 'ROOT13' in name:
                gt_type = '(√13 x √13)'
                gt_idx = TYPE_TO_IDX['(√13 x √13)']
            elif '1X1' in name or '1 X 1' in name:
                gt_type = '(1 x 1)'
                gt_idx = TYPE_TO_IDX['(1 x 1)']
            elif 'TWIN' in name or '2X1' in name:
                gt_type = 'Twinned(2 x 1)'
                gt_idx = TYPE_TO_IDX['Twinned(2 x 1)']
            elif 'C6X2' in name or 'C(6' in name:
                gt_type = 'c(6 x 2)'
                gt_idx = TYPE_TO_IDX['c(6 x 2)']
            elif 'BAD' in name:
                gt_type = 'Bad'
                gt_idx = -2  # Special marker for Bad
            else:
                gt_type = 'Unknown'
                gt_idx = -1

            # Win-rate classification
            winrate_result = classify_winrate(model, img_path, device, all_ideal_scores, bad_scores)

            if gt_type == 'Bad':
                winrate_correct = winrate_result['is_bad']
            elif gt_idx >= 0:
                winrate_correct = (gt_idx == winrate_result['predicted_idx'])
            else:
                winrate_correct = None

            winrate_results.append({
                'file': img_path.name,
                'ground_truth': gt_type,
                'predicted': winrate_result['predicted_class'],
                'correct': winrate_correct,
                'quality': winrate_result.get('quality'),
                'classification_scores': winrate_result.get('classification_scores', {})
            })

            # Argmax classification (for comparison)
            argmax_result = classify_argmax(model, img_path, device)
            if gt_idx >= 0:
                argmax_correct = (gt_idx == argmax_result['predicted_idx'])
            elif gt_type == 'Bad':
                argmax_correct = False  # Argmax can't detect Bad
            else:
                argmax_correct = None

            argmax_results.append({
                'file': img_path.name,
                'ground_truth': gt_type,
                'predicted': argmax_result['predicted_class'],
                'correct': argmax_correct,
                'scores': argmax_result['scores']
            })

        # Print comparison table
        print(f"\n{'Image':<20} {'GT':<15} {'Win-Rate':<15} {'Quality':<8} {'Argmax':<15}")
        print("-" * 75)
        for wr_r, arg_r in zip(winrate_results, argmax_results):
            wr_status = "✓" if wr_r['correct'] else "✗" if wr_r['correct'] is not None else "?"
            arg_status = "✓" if arg_r['correct'] else "✗" if arg_r['correct'] is not None else "?"
            quality_str = f"{wr_r['quality']:.2f}" if wr_r['quality'] is not None else "N/A"
            print(f"{wr_r['file']:<20} {wr_r['ground_truth']:<15} {wr_status} {wr_r['predicted'] or 'N/A':<12} {quality_str:<8} {arg_status} {arg_r['predicted']:<12}")

        # Calculate accuracies
        valid_winrate = [r for r in winrate_results if r['correct'] is not None]
        valid_argmax = [r for r in argmax_results if r['correct'] is not None]

        if valid_winrate:
            winrate_acc = sum(1 for r in valid_winrate if r['correct']) / len(valid_winrate)
            argmax_acc = sum(1 for r in valid_argmax if r['correct']) / len(valid_argmax)
            print(f"\n  Win-Rate Accuracy: {winrate_acc:.1%} ({sum(1 for r in valid_winrate if r['correct'])}/{len(valid_winrate)})")
            print(f"  Argmax Accuracy (not principled): {argmax_acc:.1%} ({sum(1 for r in valid_argmax if r['correct'])}/{len(valid_argmax)})")

        all_results['winrate'].extend(winrate_results)
        all_results['argmax'].extend(argmax_results)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    valid_winrate_all = [r for r in all_results['winrate'] if r['correct'] is not None]
    valid_argmax_all = [r for r in all_results['argmax'] if r['correct'] is not None]

    if valid_winrate_all:
        winrate_acc = sum(1 for r in valid_winrate_all if r['correct']) / len(valid_winrate_all)
        argmax_acc = sum(1 for r in valid_argmax_all if r['correct']) / len(valid_argmax_all)
        print(f"\nWin-Rate (principled):    {winrate_acc:.1%} ({sum(1 for r in valid_winrate_all if r['correct'])}/{len(valid_winrate_all)})")
        print(f"Argmax (not principled):  {argmax_acc:.1%} ({sum(1 for r in valid_argmax_all if r['correct'])}/{len(valid_argmax_all)})")

    return all_results


def evaluate_pairwise_mode(model, device, test_split=0.2, random_seed=42):
    """
    Evaluate model in pairwise comparison mode.
    Uses held-out pairwise comparison data.
    """
    import random

    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISON EVALUATION")
    print("=" * 60)

    # Load pairwise data
    pairwise_csv = PAIRWISE_CSV if PAIRWISE_CSV.exists() else CLASSIFIER2_ROOT / "Quantum Label Data - Pairwise_Comparison.csv"
    if not pairwise_csv.exists():
        print(f"Pairwise CSV not found: {pairwise_csv}")
        return

    pairwise_df = pd.read_csv(pairwise_csv)
    print(f"Loaded {len(pairwise_df)} pairwise comparisons")

    # Split by unique pairs (same as training)
    pairwise_df['pair_key'] = pairwise_df.apply(
        lambda r: tuple(sorted([r['Image1_Path'], r['Image2_Path']])), axis=1
    )
    unique_pairs = list(pairwise_df['pair_key'].unique())

    random.seed(random_seed)
    random.shuffle(unique_pairs)
    split_idx = int(len(unique_pairs) * (1 - test_split))
    test_pairs = set(unique_pairs[split_idx:])

    test_df = pairwise_df[pairwise_df['pair_key'].isin(test_pairs)]
    print(f"Test set: {len(test_pairs)} pairs, {len(test_df)} comparisons")

    # Evaluate
    results_by_type = {rt: {'correct': 0, 'total': 0} for rt in RECONSTRUCTION_TYPES}
    results_by_winner = {'1': {'correct': 0, 'total': 0},
                         '2': {'correct': 0, 'total': 0},
                         'tie': {'correct': 0, 'total': 0},
                         'not_apply': {'correct': 0, 'total': 0}}

    transform = get_transform(training=False)

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

            results_by_type[rt]['total'] += 1
            results_by_winner[winner]['total'] += 1
            if correct:
                results_by_type[rt]['correct'] += 1
                results_by_winner[winner]['correct'] += 1

    # Print results
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

    print("\n--- Accuracy by Winner Type ---")
    for winner_type in ['1', '2', 'tie', 'not_apply']:
        stats = results_by_winner[winner_type]
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            label = {'1': 'image1 wins', '2': 'image2 wins'}.get(winner_type, winner_type)
            print(f"  {label}: {acc:.1f}% ({stats['correct']}/{stats['total']})")


def classify_single_image(model, image_path, device, all_ideal_scores=None, bad_scores=None):
    """
    Classify a single image and print detailed results.
    """
    if all_ideal_scores is None:
        all_ideal_scores = load_all_ideal_scores(model, device)
    if bad_scores is None:
        bad_scores = load_bad_image_scores(model, device)

    result = classify_winrate(model, image_path, device, all_ideal_scores, bad_scores)

    print(f"\nClassification Result for: {image_path}")
    print("-" * 50)
    print(f"Predicted Class: {result['predicted_class']}")
    if result.get('quality') is not None:
        print(f"Quality Score: {result['quality']:.2f} (percentile among ideal refs)")

    print(f"\nRaw Scores:")
    for rt, score in result['raw_scores'].items():
        print(f"  {rt}: {score:.3f}")

    print(f"\nCross-type Win-Rates:")
    for rt, wr in result.get('classification_scores', {}).items():
        if wr is not None:
            indicator = " <-- PREDICTED" if rt == result['predicted_class'] else ""
            print(f"  {rt}: {wr:.3f}{indicator}")

    if result.get('is_bad'):
        print(f"\n⚠️  Image classified as BAD (confidence: {result['bad_confidence']:.2f})")

    return result


def evaluate_ideal_split(model, device, test_fraction=0.2, exclude_twinned=True, random_seed=42):
    """
    Evaluate by splitting ideal images into reference/test sets.

    This gives us a proper 4-class (or 5-class) accuracy metric
    where we have ground truth labels for all classes.

    Args:
        model: Trained model
        device: torch device
        test_fraction: Fraction of ideal images to use for testing
        exclude_twinned: If True, exclude Twinned class (only 6 samples)
        random_seed: Random seed for reproducibility
    """
    import random

    print("\n" + "=" * 60)
    print("IDEAL SPLIT EVALUATION (Reference vs Test)")
    print("=" * 60)

    transform = get_transform(training=False)
    random.seed(random_seed)

    # Collect all ideal images by type
    ideal_images = {}
    for rec_type, ideal_dir in IDEAL_DIRS.items():
        if exclude_twinned and 'Twinned' in rec_type:
            continue
        if ideal_dir.exists():
            images = []
            for ext in ['*.png', '*.bmp']:
                images.extend(list(ideal_dir.glob(ext)))
            if images:
                ideal_images[rec_type] = sorted(images)

    # Split into reference and test
    reference_images = {}
    test_images = {}

    print(f"\nSplit (test_fraction={test_fraction}):")
    for rec_type, images in ideal_images.items():
        random.shuffle(images)
        n_test = max(1, int(len(images) * test_fraction))
        test_images[rec_type] = images[:n_test]
        reference_images[rec_type] = images[n_test:]
        print(f"  {rec_type}: {len(reference_images[rec_type])} ref, {len(test_images[rec_type])} test")

    # Compute reference scores
    reference_scores = {t: [] for t in reference_images.keys()}
    for rec_type, images in reference_images.items():
        for img_path in images:
            img = Image.open(img_path).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                scores = model(img_tensor).squeeze().cpu().numpy()
            reference_scores[rec_type].append(scores)

    # Evaluate test images using win-rate
    print(f"\n{'='*60}")
    print("Test Results by Class")
    print(f"{'='*60}")

    results_by_class = {t: {'correct': 0, 'total': 0, 'details': []} for t in test_images.keys()}
    all_correct = 0
    all_total = 0

    for gt_type, images in test_images.items():
        gt_idx = TYPE_TO_IDX[gt_type]

        for img_path in images:
            img = Image.open(img_path).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                test_scores = model(img_tensor).squeeze().cpu().numpy()

            # Compute cross-type win-rates
            classification_scores = {}
            for rec_type in reference_scores.keys():
                type_idx = TYPE_TO_IDX[rec_type]

                # Get OTHER-type references
                opponents = []
                for other_type, scores_list in reference_scores.items():
                    if other_type != rec_type:
                        opponents.extend(scores_list)

                if opponents:
                    win_rate = compute_win_rate(test_scores, opponents, type_idx)
                    classification_scores[rec_type] = win_rate

            # Predict
            if classification_scores:
                pred_type = max(classification_scores, key=classification_scores.get)
                pred_idx = TYPE_TO_IDX[pred_type]
            else:
                pred_type = None
                pred_idx = -1

            correct = (pred_idx == gt_idx)
            results_by_class[gt_type]['total'] += 1
            all_total += 1
            if correct:
                results_by_class[gt_type]['correct'] += 1
                all_correct += 1

            results_by_class[gt_type]['details'].append({
                'file': img_path.name,
                'predicted': pred_type,
                'correct': correct,
                'win_rates': classification_scores
            })

    # Print results by class
    print(f"\n{'Class':<20} {'Accuracy':<15} {'Correct/Total'}")
    print("-" * 50)
    for rec_type in sorted(results_by_class.keys()):
        stats = results_by_class[rec_type]
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"{rec_type:<20} {acc:>6.1f}%         {stats['correct']}/{stats['total']}")

    print("-" * 50)
    if all_total > 0:
        overall_acc = 100 * all_correct / all_total
        print(f"{'OVERALL':<20} {overall_acc:>6.1f}%         {all_correct}/{all_total}")

    # Print detailed results
    print(f"\n{'='*60}")
    print("Detailed Results")
    print(f"{'='*60}")
    print(f"\n{'Image':<30} {'GT':<15} {'Predicted':<15} {'Correct'}")
    print("-" * 70)
    for rec_type in sorted(results_by_class.keys()):
        for detail in results_by_class[rec_type]['details']:
            status = "✓" if detail['correct'] else "✗"
            print(f"{detail['file']:<30} {rec_type:<15} {detail['predicted'] or 'N/A':<15} {status}")

    # Return results
    return {
        'overall_accuracy': all_correct / all_total if all_total > 0 else 0,
        'by_class': results_by_class,
        'n_classes': len(test_images),
        'excluded_twinned': exclude_twinned
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Classifier2 model')
    parser.add_argument('--mode', choices=['single', 'pairwise', 'both', 'classify', 'ideal-split'],
                        default='both', help='Evaluation mode')
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    parser.add_argument('--image', type=str, help='Path to image for single classification')
    parser.add_argument('--image1', type=str, help='Path to first image for pairwise')
    parser.add_argument('--image2', type=str, help='Path to second image for pairwise')
    parser.add_argument('--type', type=str, help='Reconstruction type for pairwise comparison')
    parser.add_argument('--test-fraction', type=float, default=0.2, help='Fraction of ideals for testing')

    args = parser.parse_args()

    # Load model
    model, device = load_model(args.model)
    print(f"Model loaded. Device: {device}")

    if args.mode == 'classify' and args.image:
        # Classify single image
        classify_single_image(model, args.image, device)
        return

    if args.mode == 'ideal-split':
        evaluate_ideal_split(model, device, test_fraction=args.test_fraction)
        return

    if args.mode in ['single', 'both']:
        evaluate_single_image_mode(model, device)

    if args.mode in ['pairwise', 'both']:
        evaluate_pairwise_mode(model, device)


if __name__ == '__main__':
    main()
