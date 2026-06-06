#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the v5.7-trained Classifier2 model.

Mirrors evaluate.py but points at v5.7 CSVs and the v5.7 artifacts dir, and
wraps PIL.Image.open with a retry to tolerate iCloud Drive cold-fetches.
"""

import sys
import time
from pathlib import Path
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))

import train_unified as tu

tu.PAIRWISE_CSV = tu.CLASSIFIER2_ROOT / "Quantum Label Data - Pairwise_Comparisonv5.7.csv"
tu.ABSOLUTE_CSV = tu.CLASSIFIER2_ROOT / "Quantum Label Data - Absolute_Scoringv5.7.csv"

import evaluate as ev

ev.PAIRWISE_CSV = tu.PAIRWISE_CSV
ev.ABSOLUTE_CSV = tu.ABSOLUTE_CSV

ARTIFACTS_DIR = tu.CLASSIFIER2_ROOT / "artifacts_v5.7"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"

_orig_open = PILImage.open

def _open_with_retry(fp, *args, **kwargs):
    last_err = None
    for attempt in range(5):
        try:
            return _orig_open(fp, *args, **kwargs)
        except (OSError, TimeoutError) as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise last_err

PILImage.open = _open_with_retry


def main():
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Loading model: {MODEL_PATH}")
    print(f"Device: {device}")

    model = tu.BradleyTerryModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    print("\n" + "#" * 60)
    print("# 1. IDEAL-SPLIT EVALUATION (4-class, Twinned excluded)")
    print("#" * 60)
    ev.evaluate_ideal_split(model, device, test_fraction=0.2, exclude_twinned=True)

    print("\n" + "#" * 60)
    print("# 2. PAIRWISE HOLDOUT EVALUATION (v5.7 split)")
    print("#" * 60)
    ev.evaluate_pairwise_mode(model, device, test_split=0.2, random_seed=42)

    print("\n" + "#" * 60)
    print("# 3. SINGLE-IMAGE EVAL on Test/ and Val/ folders")
    print("#" * 60)
    ev.evaluate_single_image_mode(model, device)


if __name__ == '__main__':
    main()
