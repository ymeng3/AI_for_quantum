# Project Quantum — AI-Guided RHEED Characterization & MBE Growth Control

This repo hosts the code, labeling tool, and documentation for an NSF-funded project on automating MBE growth control via real-time RHEED image analysis (**Stage A — Classifier**, done) and goal-conditioned reinforcement learning (**Stage B — Control**, in design).

## What this repo does

**Stage A — Classifier (done).** A Bradley-Terry pairwise reward model that takes a single RHEED frame and outputs a 5-dimensional reward vector across 5 reconstruction types ((1×1), Twinned(2×1), c(6×2), (√13×√13), HTR), then classifies via cross-type win-rate against ideal reference images and produces a same-type quality percentile. A "Bad" class is detected by comparison against bad reference images.

**Stage B — Control (in design).** A goal-conditioned RL agent that consumes an expert recipe + real-time instructions, uses the Stage-A classifier as both state encoder and reward signal, and outputs growth-condition control actions. See [`Classifier2/INTERN_PROJECT_DESCRIPTION.md`](Classifier2/INTERN_PROJECT_DESCRIPTION.md) for the full plan.

## Where things live

| What | Location |
|---|---|
| **Main classifier code** | [`Classifier2/`](Classifier2/) — `train_unified.py`, `train_v5.7.py`, `evaluate.py`, `evaluate_v5.7.py` |
| **SimCLR encoder weights** (Stage 1, pretrained) | GitHub Releases — tag `v5.7-classifier` |
| **Trained reward-head weights** (Stage 2) | GitHub Releases — tag `v5.7-classifier` |
| **Labeling app** | [`labeling_software/`](labeling_software/) — Flask app, live at **<https://ai-for-quantum.onrender.com>** |
| **Pairwise & absolute-scoring labels** | [`Classifier2/Quantum Label Data - Pairwise_Comparisonv5.7.csv`](Classifier2/) and `…Absolute_Scoringv5.7.csv` (tracked in git) |
| **Image data** (ideals, trajectories) | **Google Drive** — too large for git; see `google_drive_images.json` for the file → URL mapping used by the labeling app |
| **Time-temperature logs** | [`time_temperature.csv`](time_temperature.csv) — frame timestamp → temperature, used by Stage B |

## Data layout

| Data | Count | Where | Used for |
|---|---|---|---|
| Pairwise comparisons | 996 / 244 unique pairs / 413 unique images | Git (CSV) | Stage A training |
| "Bad" labels | 31 | Git (CSV) | Stage A training (negative anchors) |
| Ideal reference images | 147 (HTR 29 / √13 31 / 1×1 41 / c(6×2) 42 / Twinned 4) | Google Drive | Stage A anchors + inference references |
| 2022 trajectory frames | 1,124 (across 3 sessions) | Google Drive | Anchoring negatives in training |
| 2025 trajectory frames | 3,400+ (growing) | Google Drive | **Stage B** (not yet used in Stage A) |
| Test / Val held-out | 12 (HTR + RT13 only) | Google Drive | Held-out single-image evaluation |
| Model checkpoints | 2 (`best_model.pth`, `final_model.pth`) | GitHub Release | Inference / fine-tuning |

## What data was dropped or is not used (and why)

| Item | Status | Reason |
|---|---|---|
| 10 RT13 ideals (`RT13_1` … `RT13_10`) | **Removed** from ideal set | 8 had bright backgrounds from inverted-polarity capture; pixel-level fixes (inversion, brightness scaling) destroyed the diffraction pattern. The original captures don't contain a usable RHEED pattern. See [`Classifier2/PROGRESS.md`](Classifier2/PROGRESS.md). |
| 3 HTR ideals (`HTR_17, 18, 21` `.png`) | **Removed**, replaced | Newer `.bmp` versions of HTR_20–24 added as cleaner replacements |
| 3 borderline 1×1 ideals (`1x1_25, 26, 27`) | **Flagged, not yet removed** | All three predict as HTR with quality ≈ 0.08, all five raw scores negative, win-rates near-tied between 1×1 and HTR. Likely one bad capture session — needs expert review. |
| 2025 trajectory frames (~3,400) | **Not used in training (yet)** | Currently being collected; reserved for Stage B RL training |
| "Other" reconstruction type rows (9 in v5.7) | Filtered at training time | Not one of the 5 known reconstruction types |
| Twinned(2×1) ideals (4 images) | Used as anchors, excluded from formal ideal-split eval | Sample size too small for a meaningful 20% holdout |
| `Confidence` column in pairwise CSV | Absent in v5.7 export (was present in v1.8) | Export schema changed. Training code's confidence-weighting path is intact but currently inactive (defaults to weight = 1.0). |
| Test / Val folders (12 images) | Not used in training | Explicit held-out evaluation only |

## Quick start

```bash
git clone https://github.com/ymeng3/Quantum.git
cd Quantum

# Download model weights from the GitHub Releases page (tag v5.7-classifier)
# Place encoder at:        Classifier1/artifacts/encoders/simclr_resnet18_encoder.pth
# Place classifier at:     Classifier2/artifacts_v5.7/best_model.pth

# Download image data from Google Drive (link in google_drive_images.json) and
# unpack so that the data/ folder mirrors:
#   data/STO_ideal_HTR/, data/STO_ideal_RT13/, …
#   data/Trajectories/2022-02-04/, …, data/Trajectories/2025-10-04/, …
#   data/Test/, data/Val/

# Train (v5.7 config — uses the CSVs in this repo)
python3 Classifier2/train_v5.7.py

# Evaluate
python3 Classifier2/evaluate_v5.7.py
```

## Performance (v5.7)

| Evaluation | Result |
|---|---|
| Pairwise holdout (186 comparisons) | **89.8%** |
| 4-class ideal-split (27 images, Twinned excluded) | **92.6%** — HTR 100%, c(6×2) 100%, 1×1 75%, √13 100% |
| Baseline: SimCLR encoder + win-rate, no pairwise training | ~35-40% |

## Where to read more

| Document | Purpose |
|---|---|
| [`Classifier2/INTERN_ONBOARDING.md`](Classifier2/INTERN_ONBOARDING.md) | Walking-guide for a new collaborator (5 min/topic) |
| [`Classifier2/CLASSIFIER2_TECHNICAL_GUIDE.md`](Classifier2/CLASSIFIER2_TECHNICAL_GUIDE.md) | Algorithm-level deep-dive |
| [`Classifier2/PROGRESS.md`](Classifier2/PROGRESS.md) | Engineering log + design history (incl. what didn't work — NMF, VLM-as-judge) |
| [`Classifier2/INTERN_PROJECT_DESCRIPTION.md`](Classifier2/INTERN_PROJECT_DESCRIPTION.md) | Project description for the Summer 2026 intern (covers Stage B) |

## Contact

Yang Meng (CS PhD, UChicago) — `ymeng3@uchicago.edu` — primary mentor.
