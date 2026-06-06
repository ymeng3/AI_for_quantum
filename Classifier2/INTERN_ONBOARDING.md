# Intern Onboarding — Walking Guide

A walking guide for three topics — paths, numbers, and the key talking points so each can be covered in ~5 minutes.

## 1. Data — what we have & where it lives

Everything lives in the repo **[github.com/ymeng3/AI_for_quantum](https://github.com/ymeng3/AI_for_quantum)**; paths below are relative to the repo root.

| Data | Location | Count | Notes |
|---|---|---|---|
| **Ideal reference images (5 classes)** | `data/STO_ideal_HTR/`, `STO_ideal_RT13/`, `STO_ideal_1x1/`, `STO_ideal_c6x2/`, `STO_ideal_Twinned2x1/` | **29 + 31 + 41 + 42 + 4 = 147** | Gold-standard examples per class; used both as training anchors and as references at inference |
| **2022 trajectory frames** | `data/Trajectories/2022-02-04/`, `2022-02-06/`, `2022-04-11/` | 1,124 | Old setup (AJ era), used as anchoring negatives in training |
| **2025 trajectory frames** | `data/Trajectories/2025-10-04/{A,B}/`, `2025-10-05/` | 3,400+ | New setup, **actively being collected** — filenames carry temp + timestamp |
| **Test / Val holdout** | `data/Test/`, `data/Val/` | 12 total | Small, only HTR + RT13 |
| **Pairwise labels** | `Classifier2/Quantum Label Data - Pairwise_Comparisonv5.7.csv` | 996 comparisons / 244 unique pairs / 413 images | Exported from labeling app; main training data for Stage A |
| **Bad labels** | `Classifier2/Quantum Label Data - Absolute_Scoringv5.7.csv` | 31 bad | Negative anchors |
| **Time–temp logs** | `time_temperature.csv` (repo root) | — | Per-frame temperature; Stage B will use |
| **Trained classifier weights** | `Classifier2/artifacts_v5.7/best_model.pth` | — | Stage A — ready to load |
| **SimCLR encoder** | `Classifier1/artifacts/encoders/simclr_resnet18_encoder.pth` | — | The pretrained ResNet-18 backbone the classifier is built on |

**Things to flag to the intern:**

- Only ~1,200 images have *human labels* — the rest of the trajectory frames are unlabeled.
- 2025 data is still growing — don't hard-code dataset sizes.
- Ideal filenames encode class: `HTR_5.png`, `1x1_22.bmp`, etc.

## 2. Labeling software

- **Live URL** — **https://ai-for-quantum.onrender.com**
- **Source** — `labeling_software/` in the repo (Flask)
- **Two modes:**
  - **Pairwise comparison** — two images side-by-side; pick winner (image1 / image2 / tie / not_apply) per reconstruction type
  - **Absolute scoring** — single image; flag quality + reconstruction (including "Bad")
- **Data flow:** browser click → SQLite (local) / PostgreSQL (Render) → mirrored to Google Sheets → we periodically export the CSVs above
- **Hosting:** Render free tier — *first load takes ~30 s* (cold start), warn them
- **Modifying it:** edit `app.py`, push to GitHub, Render auto-redeploys

## 3. Classifier logic

**One-liner:** train a **Bradley-Terry reward model** on pairwise comparisons, then classify single images by **win-rate against reference images**.

### Architecture — two stages

| Stage | What's trained | How |
|---|---|---|
| **1. SimCLR encoder** (ResNet-18) | Self-supervised on unlabeled RHEED | No labels — learns "what RHEED frames look like" |
| **2. Reward head** + fine-tune encoder | Pairwise + ideal + bad data, Bradley-Terry loss | Outputs **5 raw scores per image** (one per reconstruction type) |

Reward head architecture: `Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→5)`.

### Loss (per pairwise row, only on the relevant type dim T)

| Winner | Loss | Effect |
|---|---|---|
| `1` | `-log σ(r₁ − r₂)` | push img1 up on dim T |
| `2` | `-log σ(r₂ − r₁)` | push img2 up |
| `tie` | `\|r₁ − r₂\|` | pull them together |
| `not_apply` | `ReLU(r₁) + ReLU(r₂)` | push both negative |

Batch mix during training: **60% pairwise / 25% ideal anchoring / 15% bad anchoring**.

### Inference — why we can't just argmax

The 5 raw scores **aren't calibrated across dimensions** (each dim was trained independently). So argmax is meaningless. Instead use **win-rate against reference images**:

- **Classification (cross-type win-rate):** for each candidate type T,
  `wr_T = (1/N) · Σᵢ σ(r_T(x) − r_T(opponent_i))` where opponents = ideals of *all other* types.
  Predicted class = `argmax_T wr_T`. Interpretation: "x beats X% of non-T ideals on dim T."
- **Quality (same-type win-rate):** same formula but opponents = same-class ideals. Gives the percentile within the predicted class.
- **Bad detection:** average over dims of `P(bad_ref > x)`; if > **0.7**, label "Bad".

### Performance numbers

| Eval | Result |
|---|---|
| Pairwise holdout (186 comparisons) | **89.8%** |
| 4-class ideal-split (27 images) | **92.6%** (HTR 100, c(6×2) 100, 1×1 75, √13 100) |
| Doubles the SimCLR-only baseline (~35-40%) | — |

### Code map

| File | Purpose |
|---|---|
| [`Classifier2/train_unified.py`](Classifier2/train_unified.py) | Model, dataset, loss, training loop |
| [`Classifier2/evaluate.py`](Classifier2/evaluate.py) | All 5 eval modes; the win-rate logic lives in `classify_winrate()` |
| [`Classifier2/CLASSIFIER2_TECHNICAL_GUIDE.md`](Classifier2/CLASSIFIER2_TECHNICAL_GUIDE.md) | Long-form algorithm explanation |
| [`Classifier2/INTERN_PROJECT_DESCRIPTION.pdf`](Classifier2/INTERN_PROJECT_DESCRIPTION.pdf) | The proposal (what she's signing up for) |
| [`Classifier2/PROGRESS.md`](Classifier2/PROGRESS.md) | Full project context incl. what didn't work (NMF, VLM-as-judge) |
