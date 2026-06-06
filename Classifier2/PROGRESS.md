# Project Quantum — Progress Document

*Last updated: 2026-05-07*

This document tracks two parallel workstreams:

1. **Classifier** — single-image RHEED reconstruction classification (Classifier2, Bradley-Terry reward model). Mostly working; now data-limited.
2. **Growth control** — building a reward signal for an RL agent that controls MBE growth. This is the active research frontier.

---

## 1. Classifier

### 1.1 Where the model lives

- **Reward head (trained):** `Classifier2/artifacts/best_model.pth`
- **SimCLR encoder (pretrained backbone):** `Classifier1/artifacts/encoders/simclr_resnet18_encoder.pth`
- **Architecture:** ResNet-18 (SimCLR-pretrained on RHEED) → 512-dim → `Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→5)` → 5 reward scalars, one per reconstruction type. ~11.2M params total.

### 1.2 Data inventory (v5.7, 2026-04-03)

| Source | Count | Notes |
|---|---|---|
| Pairwise comparisons | 996 | up from 678 in v1.8 |
| Unique pairs | 244 | up from 179 |
| Unique images touched by labels | 413 | up from 302 |
| Bad images (Absolute scoring) | 31 | up from 18 |
| Ideal HTR | 21 (was 19) | `data/STO_ideal_HTR/` |
| Ideal RT13 (√13×√13) | 21 (after removing 8 problematic + 3 mislabeled) | originally 38; see §1.5 |
| Ideal 1×1 | 27 | `data/STO_ideal_1x1/` |
| Ideal c(6×2) | 24 | `data/STO_ideal_c6x2/` |
| Ideal Twinned(2×1) | 6 | `data/STO_ideal_Twinned2x1/` — small sample, excluded from formal eval |
| 2022 trajectory frames | 1,124 | `Trajectories/2022-02-04`, `2022-02-06`, `2022-04-11` |
| 2025 trajectory frames | ~3,400+ | `Trajectories/2025-10-04/{A,B}`, `2025-10-05` |

**Important data quality note:** v5.7 dropped the `Confidence` and brightness columns that were present in v1.8 (313 confident + 15 somewhat-sure labels). Confidence weighting in `train_unified.py` (lines 86-91, 188-190) currently has no signal in v5.7. Either re-export with `Confidence` retained, or accept default weight = 1.0 going forward.

### 1.3 Training recipe (`Classifier2/train_unified.py`)

Each batch is a mixture:
- **60% pairwise comparisons** — Bradley-Terry loss with confidence weighting
- **25% ideal anchoring** — ideal vs random trajectory (70%) or ideal vs other-type ideal (30%)
- **15% bad-image filtering** — bad image vs random trajectory, bad must lose

Bradley-Terry loss (per row, weighted by confidence `w`):
- winner=1 → `w · −log σ(r₁ − r₂)`
- winner=2 → `w · −log σ(r₂ − r₁)`
- tie → `w · |r₁ − r₂|`
- not_apply → `w · (ReLU(r₁) + ReLU(r₂))` (push both negative)

Hyperparameters: AdamW, lr=1e-4, wd=1e-4, batch=16, 30 epochs, cosine LR, grad clip 1.0, 500 samples/epoch. Train/test split is at the **pair level** (80/20) so all comparisons for a given pair stay in one split.

### 1.4 Evaluation (`Classifier2/evaluate.py`)

The model outputs 5 reward scores, but those scores **are not calibrated across dimensions** — naive argmax fails because each dimension was trained independently with the Bradley-Terry objective. Inference uses **win-rate against reference images** instead:

- **Classification (cross-type win-rate):** for each candidate type T, compute `(1/N) · Σ σ(r_T(x) − r_T(opponent_i))` over ideal images of *other* types. Pick the T with highest win-rate. This is the only principled way to compare across dimensions — it converts the per-dim reward into "fraction of non-T references this image beats on dim T."
- **Quality score (same-type win-rate):** `(1/N) · Σ σ(r_T(x) − r_T(same-type ref_i))` — percentile among ideal references of the predicted class.
- **Bad detection:** compare against bad references on every dimension; if bad refs typically win, classify as Bad (threshold 0.6).

**Why win-rate uses sigmoid-then-mean, not mean-then-sigmoid:** σ is non-linear, so order matters. Mean-then-sigmoid lets one extreme reference bias the result; sigmoid-then-mean bounds each reference's influence to [0,1] and is the proper aggregation of Bradley-Terry probabilities.

### 1.5 Data cleaning that has happened

#### RT13 ideal cleanup (worknote 3.1)

Original ideal RT13 set was 38 images. 8 of them (`RT13_1, 3, 5, 6, 7, 8, 9, 10`) had **bright backgrounds** (mean pixel ~230-236) instead of the dark backgrounds (~6-17) that the rest had. The classifier was confusing these with HTR (which is naturally bright, mean ~230-240). Result: RT13 accuracy dropped to 42.9%.

Two fix attempts that **failed**:

| Attempt | What we tried | Why it failed |
|---|---|---|
| Inversion (`255 - pixel`) | Flip pixel values to dark-background | Images are RGB with green phosphor — inversion flips all color channels, destroying the RHEED pattern. Resulting images were mostly black with no visible diffraction spots. |
| Brightness scaling (× 0.05) | Multiply values down to mean ~12 | So aggressive that all pattern detail was lost — images went nearly completely black. |

**Why nothing pixel-level works for these 8:** the bright-background images were photographed with **inverted polarity at acquisition** (different phosphor screen state). The pattern is genuinely faint against the bright background. When you invert, the low-contrast pattern is preserved as the math says, but the dots are now invisible against the new dark background. There is no way to recover what the camera did not capture — these images simply don't contain a usable RHEED pattern (overexposed, out of focus, or pattern blocked by beam stop).

**Decision:** removed the 8 bright RT13 images + 3 mislabeled-as-c6x2. Final ideal RT13 set is 21 clean images. *A physics-aware correction (inverting only luminance in LAB/HSV while preserving hue) was considered but not pursued — removal was simpler and the data was bad regardless.*

#### Remaining classifier failure cases (worknote 3.1)

After cleaning, two failure modes remain on 1×1 test images:

| Image | True | Predicted | Quality | Why |
|---|---|---|---|---|
| `1x1_25.bmp` | 1×1 | HTR | 0.09 | Win-rate and softmax disagree; very low quality → genuinely ambiguous, possibly mislabeled |
| `1x1_27.bmp` | 1×1 | HTR | 0.09 | Same as above |
| `1x1_22.bmp` | 1×1 | c(6×2) | 0.41 | Both methods agree on c6x2 (63.1% softmax) — image itself is ambiguous between 1×1 and c6x2 |
| `1x1_13.bmp` | 1×1 | 1×1 | 0.47 | Correct but only 74% confident, 24.7% c6x2 — borderline |

**Two levers for future improvement:**
1. **Domain-expert review** of `1x1_25` and `1x1_27` — likely need removal, like the RT13 cleanup.
2. **More 1×1 vs c6×2 pairwise labels** — this is the boundary the model has not seen enough of. v5.7 added 63 more 1×1 and 65 more c6×2 comparisons, which should help.

### 1.6 Performance

| Evaluation | Result |
|---|---|
| Pairwise holdout (v1.8) | 90.6% overall |
| 4-class ideal split (Twinned excluded, v1.8) | 82.4% (14/17) — HTR 100%, c6x2 100%, 1×1 80%, RT13 50% |
| Baseline: SimCLR + Prototype | 41.2% |
| Baseline: SimCLR + Win-Rate (no pairwise training) | 35.3% |

Pairwise training **roughly doubles** single-image accuracy. The learning curve (`learning_curve.py`) at fractions {20, 40, 60, 80, 100}% shows monotone improvement (68.8 → 75.0 → 75.0 → 81.2 → 83.3) with no plateau — **model is data-limited, not architecture-limited**. v5.7's +318 comparisons should push this higher; needs retraining + re-eval.

### 1.7 Insight worth claiming as novelty (meeting 1.19)

The win-rate inference scheme — reference-anchored Bradley-Terry, where pairwise-trained per-dimension rewards are made cross-comparable via reference images — appears to be a non-standard formulation. Worth exploring connections to kernel methods (it's a non-parametric similarity-style aggregation) and combining quality + classification error into a single evaluation metric for a paper write-up.

### 1.8 Open issue: false positives with low epistemic uncertainty

Discussed at meeting 1.19. The "high quality, low accuracy" failure mode (e.g. `1x1_25` confidently misclassified) maps onto the open-world / unknown-unknowns regime described in *"Identifying Unknown Unknowns in the Open World"*: the classifier confidently assigns a known label to an instance that may belong to an unmodeled or ambiguous category, so uncertainty sampling alone cannot surface it. Active learning on borderline pairs (1×1 vs c6×2 specifically) is the practical fix.

### 1.9 Classifier — next steps

- [ ] Retrain on v5.7 (+47% comparisons) and re-evaluate against v1.8 baselines
- [ ] Recover `Confidence` column in v5.7 export (or accept default 1.0)
- [ ] Domain-expert review of `1x1_25`, `1x1_27` (likely removals)
- [ ] Targeted 1×1 vs c(6×2) pairwise collection through the labeling app
- [ ] Add labeler note explanations as future LLM context (meeting 1.19)
- [ ] Consider Twinned data collection — current ideal set is only 6, excluded from formal eval

---

## 2. Growth control — toward an RL reward signal

The end goal: train an RL agent that controls MBE growth (temperature, oxygen pressure, deposition rate) to drive the surface toward a target reconstruction. The agent needs a **reward signal** at every step.

### 2.1 What we tried first: NMF on raw pixels

**Hypothesis.** RHEED frames are non-negative; if we run NMF on a stack of trajectory frames `V ≈ W·H`, the basis components `H` should correspond to physical reconstruction patterns, and the per-frame weights `W` should track phase transitions over time. This would give an unsupervised, low-dimensional, interpretable state representation — perfect for RL.

**What worked (within a single same-setup session):**
- Same-year decompositions converge cleanly (~7-10% reconstruction error)
- Same-setup HTR consistency ≈ 0.91-0.93
- Same-setup RT13 consistency ≈ 0.99
- Real temperature-correlated phase transitions are visible in W over time
- Frame-by-frame story matches physics: e.g. RR220411A at 539°C is dominated by component 1 (high-temp phase); at 741°C component 2 emerges and becomes the √13×√13 reconstruction; on cooldown component 2 strengthens.

NMF is a **solid real-time monitoring tool within a single experimental session** — that part works.

### 2.2 Why NMF failed for our actual goal

#### Failure mode 1: Pixel-space NMF picks brightness, not diffraction

NMF minimizes total pixel reconstruction error. Given 150 ideal images across 4 classes and 4 components, NMF asks "what 4 templates minimize total pixel error?" — the answer is templates that capture **brightness gradients and screen geometry**, because those account for ~95-98% of pixel variance. The diffraction spot differences between RT13 and HTR are maybe **1-2% of total pixel variance**; NMF has no reason to spend a component on that when it can reduce more error by modeling the easy stuff.

Concrete result: **HTR vs RT13 cosine similarity in NMF basis space = 0.987** — they look essentially identical to the algorithm. C1, C3, C4 all became "HTR-like" and RT13 got no dedicated component. The basis components are physically meaningful (specular, diffraction spots, etc.) but **don't separate classes**.

This is a property of NMF's objective, not a tuning issue. More components and stronger regularization don't fix it — they just split the brightness explanation into more pieces.

#### Failure mode 2: Cross-setup projection breaks

Even if the H matrix is meaningful within one setup, projecting new trajectory frames onto a previously trained H fails when the imaging conditions change.

| Comparison | Brightness mean | Brightness std | KS statistic | Cosine sim of mean image |
|---|---|---|---|---|
| 2022 alone | 14.7 | ±1.4 | — | — |
| 2025 alone | 9.7 | ±6.2 (4× more variable) | — | — |
| 2022 vs 2025 | — | — | 0.32 (p ≈ 0) | 0.93 |

The 2025 setup is ~34% darker on average **and** has 4× more frame-to-frame brightness variance — the signal sits in a different pixel regime. The 0.93 cosine similarity is misleading: most of it comes from shared background structure, not diffraction features. KS test confirms statistically very different distributions.

Cross-year basis similarity:
- HTR: 0.909 within 2025 → drops to 0.444 (HL251004A vs RR220206A 2022 vs 2025)
- RT13: 0.997 within 2022 → drops to 0.270 (RR220204A vs HL251005A)

A new 2025 trajectory frame projected onto a 2022 ideal basis yields garbage W — it's like trying to describe a red image as a combination of blue templates. The basis can't span the new pixel space.

#### Failure mode 3: Preprocessing helps RT13 but hurts HTR

Tried CLAHE, percentile normalization, and histogram equalization to align brightness distributions before projection.

| Method | RT13 cross-year sim | RT13 same-year sim | RT13 error | HTR cross-year sim | HTR same-year sim | HTR error |
|---|---|---|---|---|---|---|
| Raw | 0.248 | 0.997 | 12.9% | 0.576 | 0.909 | 15.9% |
| CLAHE | **0.586** | 0.991 | 7.1% | **0.195** | 0.927 | 8.2% |
| Percentile | 0.586 | 0.990 | 7.1% | 0.193 | 0.926 | 8.2% |
| HistEq | 0.524 | 0.958 | 5.7% | 0.203 | 0.496 | 7.2% |

CLAHE more than doubled RT13 cross-year consistency (0.248 → 0.586), but actually made HTR cross-year *worse* (0.576 → 0.195). After normalizing intensity, what's left is a **genuine structural difference** in the RHEED setup between 2022 and 2025 — a global pixel correction can't fix a different camera angle, phosphor screen state, or beam alignment. **The 2022 vs 2025 gap is a real physical setup change, not a data quality issue.**

#### Failure mode 4: Ideal-image-trained NMF doesn't transfer either

Tried training NMF only on the 99 cleaned ideal images so the H basis would be "physically meaningful" by construction, then projecting trajectory frames. Same problem reappears: H is learned to reconstruct 2022-style ideal pixels, and a 2025 trajectory frame still looks nothing like any linear combination of those.

#### Why W weights aren't temporally smooth

NMF treats each frame independently — no temporal regularization. So you get sudden jumps even when consecutive frames are nearly identical (NMF found two equally valid decompositions and picked different ones), and noisy flickering when a frame sits between two components. **Fix:** rolling-average post-processing on W if NMF is used at all. (Not a deep fix — just cosmetic for monitoring.)

### 2.3 Diagnosis

This is a **domain gap problem**, not an NMF objective problem. The 2025 frames and the 2022 ideal basis live in different pixel spaces. Even fixing NMF's objective (semi-supervised NMF à la Lee/Choi 2010, discriminative NMF à la Zafeiriou/Petrou 2010, graph-regularized variants) operates on **raw pixels** where brightness still dominates. They are incremental improvements on the same fundamental problem.

### 2.4 What should solve it

Two practical paths:

| Option | What it is | Status |
|---|---|---|
| Histogram matching → NMF | Match each trajectory frame's pixel distribution to the ideal images before projection | Tried (CLAHE) — partial success on RT13, fails on HTR |
| **Feature-space projection** | Extract features from Classifier2 (which was *trained* to be domain-invariant via cross-setup pairwise comparisons), then run NMF on those features instead of raw pixels | The principled fix |
| **Classifier directly** | Skip NMF entirely; use Classifier2 reward scores as the live signal | Recommended |

### 2.5 What we tried for VLM-as-judge (and why it didn't replace the classifier)

Idea: prompt a vision-language model with `"This is a RHEED diffraction pattern. Rate on a scale of 0-1 how closely it matches the RT13 reconstruction."` Should be able to focus on diffraction spot geometry and ignore brightness, by construction.

**What broke it:**
- **Too slow:** ~2-5 sec per VLM API call vs ~5 ms for the classifier. RL needs reward at every step (potentially millions of times per training run) — that's a 1000× speed gap, infeasible as a live reward.
- **Fails on HTR and RT13 specifically.** These are the *hard* classes — distinguishing features (1/√13 order spots, specific streak modulations, spot spacings) require genuine domain expertise. Frontier VLMs have likely seen very few RHEED examples in training, and the differences are too fine-grained for general visual reasoning. VLMs do fine on 1×1 (clean streaks) and c(6×2) (visually dramatic) — but those are exactly the classes Classifier2 already handles well.
- **Few-shot prompting won't save it.** Few-shot helps when the VLM can perceive the relevant features but doesn't know what they're called. That's not the situation here — VLMs process images as coarse patch tokens and may literally lack the spatial resolution to perceive the 1/√13 order spot differences. Showing labeled examples doesn't restore resolution that wasn't there.

| Use case | VLM | Classifier2 |
|---|---|---|
| HTR identification | ✗ Fails | ✓ |
| RT13 identification | ✗ Fails | ✓ |
| 1×1 / c6x2 | ✓ Slow | ✓ Fast |
| RL reward signal | ✗ Slow + inaccurate on hard cases | ✓ |
| VLM distillation | ✗ Can't distill knowledge VLM doesn't have | — |

**Conclusion:** VLM distillation is off the table for HTR/RT13. VLM as a *periodic check* during RL rollouts (every N episodes, sample a few frames, ask "is this trajectory making progress toward HTR?") is the only useful role.

This actually validates the whole project setup: the reason we needed expert pairwise labeling in the first place was precisely because HTR and RT13 are too subtle for general visual reasoning. Classifier2 learned exactly the expert knowledge VLMs lack.

### 2.6 The chosen path: Classifier2 as the reward

```python
scores = model(frame)              # shape [5]
reward = scores[TYPE_TO_IDX['HTR']] # scalar Bradley-Terry score for HTR
# or, bounded in [0,1]:
probs = F.softmax(scores, dim=0)
reward = probs[TYPE_TO_IDX['HTR']]
```

This already is:
- **Setup-invariant** — trained on cross-setup pairwise comparisons, not raw pixels
- **Physically meaningful** — trained on human expert judgments
- **Multi-dimensional** — all 5 scores available simultaneously (e.g. watch 1×1 score drop while HTR score rises)
- **Differentiable, no argmax** — the raw score is already a continuous signal
- **Fast** — ~5ms inference

Adding NMF on top of classifier features (Option 2 from §2.4) is interesting for *interpretability/analysis* (gradual transitions, sub-patterns within a class), but it doesn't add reward signal that isn't already in the classifier output.

### 2.7 Reward formulation with physics priors

Instead of a fixed target, the reward should track the **growth recipe** — different phases of growth target different reconstructions:

```python
reward = classifier_score[recipe_target(t, T, P_O2)]
       - β · max(0, T - T_max)            # too hot penalty
       - β · max(0, T_min - T)            # too cold penalty
       - β · recipe_phase_violation       # wrong reconstruction for current conditions
```

**To build this, we need:**
1. **Formal recipe encoding** — from the standard MBE growth recipe (probably defined at discrete checkpoints; need an interpolation function between them).
2. **Time-temperature data to calibrate the recipe** — already have `time_temperature.csv` in the repo root; need to align it with frame timestamps.

### 2.8 Trajectory data collection — what we need going forward

For each new trajectory:

**Per trajectory:**
1. Target reconstruction type
2. Outcome (success / partial / failed, what was observed)
3. Brief notes on deviations from standard recipe
4. Operator name + date

**Per frame** (automated):
1. RHEED image saved at fixed interval
2. Timestamp synchronized with process log
3. Current temperature
4. Current oxygen pressure
5. Filename convention: `{trajectory_id}_{frame_number}_{temperature}C_{time}.bmp`

**Keyframe annotations** (5-10 per trajectory, human):
1. Frame where growth clearly started
2. Frame where target reconstruction first appeared
3. Frame where it stabilized
4. Frame of any unexpected transitions
5. Quality rating of final state (good / acceptable / poor)

The 2025 trajectories (`HL251004A`, `HL251004B`, `HL251005A`, plus `2025-10-04/A` ~2,500 frames and `B` ~900 frames) follow this convention. The 2022 trajectories (`RR220204A`, `RR220206A`, `RR220411A`) are from the AJ paper era and predate this convention.

**For NMF analysis specifically, we need:** 5-10 trajectories per reconstruction type, all on the **same machine without realigning** the beam/camera/screen, with varying growth parameters. Right now: 2025 has 2 HTR + 1 RT13; 2022 has 1 HTR + 2 RT13. The gap is obvious — each setup is missing trajectories of one type, which is exactly what blocks per-setup NMF analysis.

### 2.9 Open algorithmic question: pairwise + low-rank

From meeting 1.23: there's an interesting connection between our pairwise voting mechanism and low-rank latent embedding methods (Nowak et al. on active triplet/MDS-style embeddings: arxiv 1109.3701, Jamieson activeMDS, Nowak APSD, arxiv 1910.12379). Our voting majority gives a distance proxy; Nowak-style methods reconstruct a latent space from pairwise/triplet constraints with provable query complexity bounds. NMF and voting both try to construct latent embeddings — NMF can extract eigenvalue-style components (which our voting cannot), but voting incorporates label supervision (which NMF cannot). Closing this gap — *"goal-oriented decomposition guided by pairwise labels"* — could be a publishable algorithmic contribution. Open questions: (1) how to choose anchors, (2) how to actively choose pairwise queries to label (our current labeling software is randomized — this could be activized).

### 2.10 Growth control — next steps

- [ ] Implement `reward = classifier_score[target]` reward function and integrate with RL training loop
- [ ] Encode the standard growth recipe formally; align with `time_temperature.csv` to interpolate target reconstruction over time
- [ ] Collect 5-10 same-setup trajectories per reconstruction type with consistent imaging
- [ ] Try feature-space NMF (Classifier2 features → NMF) as exploratory analysis tool, not as reward
- [ ] Implement temporal smoothing on NMF W if used for monitoring (rolling average)
- [ ] Add VLM as a periodic validation oracle during RL rollouts (not a per-step reward)
- [ ] Explore the pairwise/low-rank/NMF connection for a possible algorithmic contribution

---

## 3. Repository map (relevant files)

```
Project Quantum/
├── Classifier1/                                # SimCLR-pretrained encoder
│   └── artifacts/encoders/simclr_resnet18_encoder.pth
├── Classifier2/                                # Bradley-Terry reward model
│   ├── train_unified.py                        # main training script
│   ├── evaluate.py                             # 5 evaluation modes
│   ├── learning_curve.py                       # data-fraction sweep
│   ├── bradley_terry_model.py                  # model class (also defined in train_unified)
│   ├── analyze_data.py                         # data statistics
│   ├── fetch_pairwise_data.py                  # Google Sheets sync
│   ├── preprocess_images.py
│   ├── artifacts/best_model.pth                # current trained weights
│   ├── Quantum Label Data - Pairwise_Comparisonv5.7.csv     # 996 comparisons
│   ├── Quantum Label Data - Pairwise_Comparisonv1.8.csv     # 678 comparisons (legacy)
│   ├── Quantum Label Data - Absolute_Scoringv5.7.csv        # 31 bad images
│   ├── Quantum Label Data - Absolute_Scoringv1.8.csv        # 18 bad images (legacy)
│   ├── prev_worknote/                          # source for this doc
│   │   ├── Main worknote Quantum Material.pdf
│   │   ├── worknote 1.19 quantum.pdf
│   │   └── worknote 3.1 quantum.pdf
│   ├── CLASSIFIER2_TECHNICAL_GUIDE.md          # algorithm-level explanation
│   ├── CLASSIFIER2_REPORT.md                   # results report
│   └── README.md
├── data/
│   ├── STO_ideal_HTR/                          # 21 images
│   ├── STO_ideal_RT13/                         # 21 images (after cleanup)
│   ├── STO_ideal_1x1/                          # 27 images
│   ├── STO_ideal_c6x2/                         # 24 images
│   ├── STO_ideal_Twinned2x1/                   # 6 images
│   ├── Test/, Val/                             # holdout sets
│   └── Trajectories/
│       ├── 2022-02-04, 2022-02-06, 2022-04-11/ # 2022 same-setup
│       └── 2025-10-04/{A,B}, 2025-10-05/       # 2025 new setup
├── Labeled_data/                               # earlier CSV exports
├── labeling_software/                          # Flask web app, deployed to Render
│   └── app.py                                  # ~46k lines
├── NMF code/
│   ├── nmf_separate_groups.py                  # per-trajectory NMF
│   ├── nmf_same_setup_analysis.py              # same-setup consistency
│   ├── setup_comparison.py                     # 2022 vs 2025 gap analysis
│   ├── nmf_preprocess_comparison.py            # CLAHE/Percentile/HistEq sweep
│   ├── nmf_ideal_basis.py                      # train NMF on ideal images
│   └── nmf_results_*/                          # outputs per trajectory
├── time_temperature.csv                        # for recipe encoding
└── ...
```

---

## 4. Reading guide for collaborators

- For a first read of *what* the classifier does and *how* it's evaluated → `CLASSIFIER2_TECHNICAL_GUIDE.md`
- For *results numbers* → `CLASSIFIER2_REPORT.md`
- For *training code* → `train_unified.py`
- For *NMF investigations and why they failed* → §2.1-2.4 of this doc, plus `NMF code/setup_comparison_results/` figures
- For *project trajectory and meeting context* → `prev_worknote/`
