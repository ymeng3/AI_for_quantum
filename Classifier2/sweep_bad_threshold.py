#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the bad-detection threshold on the v5.7-trained model.

Forward-passes every image once, caches the [5] reward vectors, then varies the
threshold over the cached scores. Reports per-threshold:
  - Test/Val classification accuracy (no false-Bad)
  - Leave-one-out Bad recall (do real bad images still trip detection?)
  - Ideal-split accuracy (no false-Bad on ideal references)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))

import train_unified as tu

tu.PAIRWISE_CSV = tu.CLASSIFIER2_ROOT / "Quantum Label Data - Pairwise_Comparisonv5.7.csv"
tu.ABSOLUTE_CSV = tu.CLASSIFIER2_ROOT / "Quantum Label Data - Absolute_Scoringv5.7.csv"

ARTIFACTS_DIR = tu.CLASSIFIER2_ROOT / "artifacts_v5.7"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"

_orig_open = PILImage.open
def _open_with_retry(fp, *args, **kwargs):
    last = None
    for i in range(5):
        try:
            return _orig_open(fp, *args, **kwargs)
        except (OSError, TimeoutError) as e:
            last = e
            time.sleep(0.5 * (i + 1))
    raise last
PILImage.open = _open_with_retry

THRESHOLDS = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def pick_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def score(model, path, device, transform):
    img = PILImage.open(path).convert('L')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).squeeze().cpu().numpy()


def winrate_on_dim(test_score, ref_scores, dim_idx):
    diffs = np.array([test_score[dim_idx] - r[dim_idx] for r in ref_scores])
    return float(np.mean(1.0 / (1.0 + np.exp(-diffs))))


def avg_bad_score(test_score, bad_refs):
    """Returns avg over dims of P(bad_ref > test) — high means image looks bad."""
    if not bad_refs:
        return 0.0
    per_dim = []
    for d in range(tu.NUM_TYPES):
        diffs = np.array([b[d] - test_score[d] for b in bad_refs])
        per_dim.append(float(np.mean(1.0 / (1.0 + np.exp(-diffs)))))
    return float(np.mean(per_dim))


def classify_with_threshold(test_score, ideal_scores_by_class, bad_refs, threshold):
    """Return predicted class, treating result as 'Bad' if avg_bad_score > threshold."""
    bad_conf = avg_bad_score(test_score, bad_refs)
    if bad_conf > threshold:
        return 'Bad', bad_conf

    # Cross-type win-rate
    best_cls, best_wr = None, -1.0
    for rec_type in tu.RECONSTRUCTION_TYPES:
        type_idx = tu.TYPE_TO_IDX[rec_type]
        opponents = []
        for other_type, refs in ideal_scores_by_class.items():
            if other_type != rec_type:
                opponents.extend(refs)
        if not opponents:
            continue
        wr = winrate_on_dim(test_score, opponents, type_idx)
        if wr > best_wr:
            best_wr, best_cls = wr, rec_type
    return best_cls, bad_conf


def gt_from_filename(name):
    n = name.upper()
    if 'HTR' in n: return 'HTR'
    if 'RT13' in n or '√13' in n or 'ROOT13' in n: return '(√13 x √13)'
    if '1X1' in n or '1 X 1' in n: return '(1 x 1)'
    if 'TWIN' in n or '2X1' in n: return 'Twinned(2 x 1)'
    if 'C6X2' in n or 'C(6' in n: return 'c(6 x 2)'
    if 'BAD' in n: return 'Bad'
    return None


def main():
    device = pick_device()
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")

    model = tu.BradleyTerryModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    transform = tu.get_transform(training=False)

    # ---- Score all ideal images ----
    ideal_scores = {t: [] for t in tu.RECONSTRUCTION_TYPES}
    ideal_paths = {t: [] for t in tu.RECONSTRUCTION_TYPES}
    for rec_type, ideal_dir in tu.IDEAL_DIRS.items():
        if ideal_dir.exists():
            for ext in ['*.png', '*.bmp']:
                for p in sorted(ideal_dir.glob(ext)):
                    s = score(model, p, device, transform)
                    ideal_scores[rec_type].append(s)
                    ideal_paths[rec_type].append(p)
    n_ideal = sum(len(v) for v in ideal_scores.values())
    print(f"Scored {n_ideal} ideal images: " + ", ".join(f"{t}={len(s)}" for t, s in ideal_scores.items()))

    # ---- Score all bad images ----
    bad_scores = []
    bad_paths = []
    abs_df = pd.read_csv(tu.ABSOLUTE_CSV)
    bad_df = abs_df[abs_df['Reconstruction'].str.contains('Bad', na=False)]
    for _, row in bad_df.iterrows():
        p = tu.DATA_ROOT / row['File_Path']
        if p.exists():
            bad_scores.append(score(model, p, device, transform))
            bad_paths.append(p)
    print(f"Scored {len(bad_scores)} bad reference images")

    # ---- Score Test and Val images ----
    testval = []  # list of (folder, name, gt, scores)
    for folder in ['Test', 'Val']:
        d = tu.DATA_ROOT / folder
        if d.exists():
            for p in sorted(list(d.glob('*.png')) + list(d.glob('*.bmp'))):
                gt = gt_from_filename(p.stem)
                if gt is None:
                    continue
                testval.append((folder, p.name, gt, score(model, p, device, transform)))
    print(f"Scored {len(testval)} Test/Val images")

    # ---- Sweep thresholds ----
    print("\n" + "=" * 90)
    print(f"{'Thr':>5} | {'Test/Val acc':>16} | {'Bad recall (LOO)':>17} | {'Ideal-split acc':>17} | Notes")
    print("-" * 90)

    for thr in THRESHOLDS:
        # Test/Val accuracy
        tv_correct = 0
        tv_bad_fp = 0  # false-Bad (real class flagged as Bad)
        for folder, name, gt, s in testval:
            pred, bc = classify_with_threshold(s, ideal_scores, bad_scores, thr)
            if pred == gt:
                tv_correct += 1
            elif pred == 'Bad' and gt != 'Bad':
                tv_bad_fp += 1

        # Bad recall: leave-one-out — for each bad image, do other bad refs flag it as Bad?
        bad_correct = 0
        for i, s in enumerate(bad_scores):
            others = bad_scores[:i] + bad_scores[i+1:]
            pred, _ = classify_with_threshold(s, ideal_scores, others, thr)
            if pred == 'Bad':
                bad_correct += 1

        # Ideal-split: do ideal images get false-flagged as Bad? (using all-but-self refs)
        # For simplicity, just check that no ideal image gets classified as Bad
        ideal_correct = 0
        ideal_total = 0
        for cls, scores_list in ideal_scores.items():
            for i, s in enumerate(scores_list):
                # Reference set excludes this image
                refs_minus_self = {t: ([sc for j, sc in enumerate(scores_list) if j != i] if t == cls else list(slist))
                                   for t, slist in ideal_scores.items()}
                pred, _ = classify_with_threshold(s, refs_minus_self, bad_scores, thr)
                ideal_total += 1
                if pred == cls:
                    ideal_correct += 1

        tv_acc = tv_correct / len(testval) if testval else 0
        bad_rec = bad_correct / len(bad_scores) if bad_scores else 0
        ideal_acc = ideal_correct / ideal_total if ideal_total else 0

        notes = []
        if tv_bad_fp > 0:
            notes.append(f"{tv_bad_fp} Test/Val false-Bad")
        notes_str = "; ".join(notes) if notes else ""
        print(f"{thr:>5.2f} | {tv_correct}/{len(testval)} = {tv_acc:>5.1%}    | "
              f"{bad_correct}/{len(bad_scores)} = {bad_rec:>5.1%}     | "
              f"{ideal_correct}/{ideal_total} = {ideal_acc:>5.1%}     | {notes_str}")

    # ---- Per-image detail at the most-promising threshold ----
    print("\n" + "=" * 90)
    print("Per-image detail across thresholds (Test/Val only):")
    print(f"{'Image':<22} {'GT':<15} | " + " | ".join(f"{t:>5.2f}" for t in THRESHOLDS))
    print("-" * (40 + len(THRESHOLDS) * 8))
    for folder, name, gt, s in testval:
        row = [f"{folder}/{name}"]
        row.append(gt[:13])
        for thr in THRESHOLDS:
            pred, _ = classify_with_threshold(s, ideal_scores, bad_scores, thr)
            mark = "✓" if pred == gt else ("B" if pred == 'Bad' else "✗")
            row.append(f"  {mark}  ")
        print(f"{row[0]:<22} {row[1]:<15} | " + " | ".join(f"{c:>5}" for c in row[2:]))


if __name__ == '__main__':
    main()
