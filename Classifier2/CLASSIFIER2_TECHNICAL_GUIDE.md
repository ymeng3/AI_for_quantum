# Classifier2: Complete Technical Explanation (v2.1)

## Goal

Given a single RHEED image, output:
1. **Classification**: Which of 6 categories is it? (5 reconstruction types + Bad)
2. **Quality score**: How good an example of that type is it?

---

## The Challenge

We have **pairwise comparison labels** ("Image A is better than Image B for HTR"), but we want **single-image predictions**. The Bradley-Terry model bridges this gap.

---

## Model Architecture

```
Input Image (224 × 224 grayscale)
        │
        ▼
   ResNet-18 Encoder (pretrained with SimCLR on RHEED)
        │
        ▼
   512-dim feature vector
        │
        ▼
   Reward Head: Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→5)
        │
        ▼
   5 Reward Scores: [r₁, r₂, r₃, r₄, r₅]
   One score per reconstruction type:
   [(1×1), Twinned(2×1), c(6×2), (√13×√13), HTR]
```

**Key insight**: Each image gets 5 scores. Score r_T represents "how good is this image for type T" on an arbitrary scale learned from pairwise comparisons.

### Architecture Dimensions

| Transformation | From | To | Reason |
|---------------|------|-----|--------|
| Grayscale → RGB | 1 ch | 3 ch | Match pretrained ResNet input |
| Resize | Variable | 224×224 | Standard ImageNet size |
| ResNet-18 | 224×224×3 | 512 | Extract learned visual features |
| Hidden layer | 512 | 256 | Task-specific transformation |
| Output | 256 | 5 | One reward per reconstruction type |

### Why Grayscale Input?

RHEED images are often captured with a green tint (phosphor screen artifact). We convert to grayscale because:
- The color is noise, not signal - physics information is in intensity patterns
- Ensures consistency across different cameras/setups
- The SimCLR encoder was pretrained on grayscale RHEED images
- Prevents model from learning spurious color correlations

---

## Training Data Sources (v1.8)

### 1. Pairwise Comparisons (Primary - 60% of batches)

**678 total comparisons** across 179 unique image pairs.

Human labels of the form:
- Image1, Image2, Type=HTR, Winner=1 → "Image1 is better HTR than Image2"
- Image1, Image2, Type=RT13, Winner=2 → "Image2 is better RT13 than Image1"
- Image1, Image2, Type=c(6×2), Winner=tie → "Both equally good for c(6×2)"
- Image1, Image2, Type=HTR, Winner=not_apply → "Neither is HTR"

**Distribution by type**:
| Reconstruction Type | Comparisons |
|--------------------|-------------|
| (√13 x √13) | 157 |
| (1 x 1) | 140 |
| HTR | 137 |
| c(6 x 2) | 120 |
| Twinned(2 x 1) | 115 |

**Distribution by winner**:
| Winner | Count |
|--------|-------|
| not_apply | 262 |
| Image 2 wins | 169 |
| Image 1 wins | 158 |
| tie | 89 |

### 2. Ideal Reference Anchoring (25% of batches)

Reference images for ALL 5 reconstruction types:

| Type | Folder | Count |
|------|--------|-------|
| HTR | `STO_ideal_HTR` | 21 |
| (√13 x √13) | `STO_ideal_RT13` | 21 |
| (1 x 1) | `STO_ideal_1x1` | 27 |
| Twinned(2 x 1) | `STO_ideal_Twinned2x1` | 6 |
| c(6 x 2) | `STO_ideal_c6x2` | 24 |

**Synthesized pairs from references**:
```
ideal_HTR vs random_trajectory on HTR dim → ideal wins
ideal_1x1 vs ideal_RT13 on 1x1 dim → 1x1 wins
ideal_T vs ideal_other on T dim → T wins
```

### 3. Bad Image Filtering (15% of batches)

**18 bad images** from `Quantum Label Data - Absolute_Scoringv1.8.csv` marked as "Bad".

```
bad_image vs any_trajectory on any dim → bad loses
bad_image on any dim → push score negative
```

### 4. Confidence Weighting (New in v2.0)

Some pairwise labels include confidence information:
- **Confident**: weight = 1.0
- **Somewhat sure**: weight = 0.7
- **No confidence data**: weight = 1.0 (default)

Higher-confidence labels have more influence on the loss.

---

## Loss Functions

### Bradley-Terry Loss

The probability that Image A beats Image B on dimension T:

```
P(A > B | type T) = σ(r_T(A) - r_T(B)) = 1 / (1 + exp(-(r_T(A) - r_T(B))))
```

**Loss by winner outcome** (with confidence weighting w):

| Winner | Loss | Effect |
|--------|------|--------|
| Image 1 wins | w × -log σ(r₁ - r₂) | Push r₁ > r₂ |
| Image 2 wins | w × -log σ(r₂ - r₁) | Push r₂ > r₁ |
| Tie | w × \|r₁ - r₂\| | Push scores together |
| Not applicable | w × (ReLU(r₁) + ReLU(r₂)) | Push both negative |

---

## What Training Learns (and Doesn't Learn)

For each dimension T, the model learns a **relative ranking** of images:
- Images that frequently win on dimension T get higher r_T
- Images that frequently lose get lower r_T

**Critical limitation**: Dimensions are learned **independently**. There's no training signal comparing across dimensions. The scales are arbitrary and not calibrated to each other.

**Why argmax doesn't work**:
```
Image X: r_HTR = 3.0, r_RT13 = 2.5
```
You cannot conclude "X is HTR" because:
- HTR dimension might use scale [-2, +5]
- RT13 dimension might use scale [-1, +3]
- 3.0 on HTR scale ≠ 2.5 on RT13 scale (incomparable)

---

## Inference: Win-Rate Method

### The Key Insight

We can't compare scores across dimensions directly, but we CAN compare any two images on the SAME dimension. This is exactly what Bradley-Terry was trained to do.

**Solution**: Use reference images to calibrate each dimension via **win-rate**.

### Win-Rate Formula

Instead of comparing against the mean (naive approach):
```
P = σ(r_T(x) - mean(r_T(refs)))  ← NAIVE: mean-then-sigmoid
```

We compute win-rate against each reference individually:
```
WinRate_T(x) = (1/N) Σᵢ σ(r_T(x) - r_T(ref_i))  ← CORRECT: sigmoid-then-mean
```

**Why this is better**:

| Aspect | Mean-then-Sigmoid | Win-Rate (Sigmoid-then-Mean) |
|--------|-------------------|------------------------------|
| Outlier handling | One extreme ref biases result | Each ref has bounded influence [0,1] |
| Interpretation | "Beat some aggregate" | "Beat X% of references" |
| Mathematical validity | σ is nonlinear, order matters | Proper aggregation of BT probabilities |

---

## Complete Inference Pipeline (6-Class)

### Step 1: Get Reward Scores
```
Input image x → Model → [r₁ₓ₁, r_twin, r_c62, r_rt13, r_htr]
```

### Step 2: Bad Image Detection

First, check if image is "Bad" by comparing against bad reference images:
```
For each dimension T:
  bad_score_T = (1/N_bad) Σᵢ σ(r_T(bad_ref_i) - r_T(x))

avg_bad_score = mean(bad_score_T for all T)
```

If `avg_bad_score > threshold` (e.g., 0.6), classify as **Bad**.

### Step 3: Classification (Cross-Type Win-Rate)

For each candidate type T, compute: "What fraction of OTHER-type references does x beat on dimension T?"

```
For T = HTR:
  opponents = all non-HTR ideal images (RT13, 1x1, Twinned, c6x2)
  win_rate_HTR = (1/N) Σᵢ σ(r_HTR(x) - r_HTR(opponent_i))

For T = (1 x 1):
  opponents = all non-1x1 ideal images
  win_rate_1x1 = (1/N) Σᵢ σ(r_1x1(x) - r_1x1(opponent_i))

... (same for all 5 types)

Predicted class = argmax(win_rate_HTR, win_rate_RT13, win_rate_1x1, ...)
```

**Why compare against OTHER types?**
- If x is truly HTR, it should beat non-HTR images on the HTR dimension
- This is a valid Bradley-Terry comparison (same dimension)
- The win-rate tells us "how much more HTR-like is x compared to non-HTR images?"

### Step 4: Quality Score (Same-Type Win-Rate)

After classification, compare against SAME-type references:

```
If predicted = HTR:
  quality = (1/N) Σᵢ σ(r_HTR(x) - r_HTR(HTR_ref_i))
```

**Interpretation**:
- quality = 0.90 → "Beats 90% of ideal HTR references" → excellent
- quality = 0.50 → "Average among ideal HTRs" → decent
- quality = 0.20 → "Worse than 80% of ideal HTRs" → poor quality

---

## Full Example (6-Class)

```
Test image x arrives

1. Forward pass:
   x → model → [r_1x1=1.2, r_twin=0.8, r_c62=0.5, r_rt13=1.8, r_htr=2.5]

2. Load reference scores (precomputed):
   HTR_refs:      [htr: 2.1-2.4, rt13: 0.2-0.5, 1x1: 0.3-0.6, ...]  (19 images)
   RT13_refs:     [rt13: 1.9-2.2, htr: 0.6-0.9, ...]                 (21 images)
   1x1_refs:      [1x1: 1.5-1.9, htr: 0.4-0.7, ...]                  (20 images)
   Twinned_refs:  [twin: 1.3-1.6, ...]                                (6 images)
   c6x2_refs:     [c62: 1.4-1.8, ...]                                 (17 images)
   Bad_refs:      [all dims: -1.5 to 0.3]                             (18 images)

3. Bad detection:
   Compare x against bad_refs on all dimensions
   avg_bad_score = 0.15  (x beats most bad refs → NOT bad)

4. Classification - compute win-rates against OTHER types:

   For HTR: "Does x beat non-HTR refs on HTR dimension?"
     opponents = RT13_refs + 1x1_refs + Twinned_refs + c6x2_refs (64 total)
     win_rate_HTR = (1/64) Σᵢ σ(2.5 - opponent_i[htr])
                  ≈ 0.88  (beats 88% of non-HTR refs on HTR metric)

   For RT13: "Does x beat non-RT13 refs on RT13 dimension?"
     win_rate_RT13 = 0.72

   For (1x1): win_rate_1x1 = 0.45
   For Twinned: win_rate_twin = 0.38
   For c(6x2): win_rate_c62 = 0.31

   Prediction: HTR (0.88 is highest)

5. Quality - compare to same-type refs:
   quality = (1/19) Σᵢ σ(2.5 - HTR_ref_i[htr])
           ≈ 0.57  (beats 57% of ideal HTR refs)

6. Final output:
   Class: HTR
   Quality: 0.57 (slightly above average for HTR)
   Classification confidence: 0.88 (high confidence)
   Runner-up: RT13 at 0.72
```

---

## Why Reference Images Are Essential

| Without References | With References |
|-------------------|-----------------|
| Raw scores on arbitrary scales | Calibrated win-rates (0-1) |
| Can't compare across dimensions | Can compare: "88% HTR vs 72% RT13" |
| Only pairwise comparison works | Single-image classification works |
| No quality interpretation | Quality = percentile among ideals |
| No "Bad" detection | Can detect bad images via reference comparison |

---

## Current Status & Data (v2.1)

### Ideal References (Complete!)

| Type | Folder | Count | Status |
|------|--------|-------|--------|
| HTR | `STO_ideal_HTR` | 21 | ✓ |
| (√13 x √13) | `STO_ideal_RT13` | 21 | ✓ |
| (1 x 1) | `STO_ideal_1x1` | 27 | ✓ |
| Twinned(2 x 1) | `STO_ideal_Twinned2x1` | 6 | ✓ (small sample) |
| c(6 x 2) | `STO_ideal_c6x2` | 24 | ✓ |

**Total ideal images: 99**

### Bad Images

| Source | Count |
|--------|-------|
| Absolute_Scoringv1.8.csv | 18 |

### Pairwise Data

| Source | Comparisons | Pairs | With Confidence |
|--------|-------------|-------|-----------------|
| Pairwise_Comparisonv1.8.csv | 678 | 179 | 328 |

---

## Evaluation Results (January 2025)

### 4-Class Ideal Split Evaluation

Split ideal images into 80% reference / 20% test to evaluate single-image classification:

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| HTR | 100% | 4/4 |
| c(6 x 2) | 100% | 4/4 |
| (1 x 1) | 80% | 4/5 |
| (√13 x √13) | 50% | 2/4 |
| **Overall** | **82.4%** | **14/17** |

*Note: Twinned excluded due to small sample size (6 images)*

### Pairwise Holdout Evaluation

| Metric | Result |
|--------|--------|
| Overall pairwise accuracy | 90.6% |

### Comparison to Baseline

Tested SimCLR encoder without pairwise training to verify pairwise data helps:

| Method | Accuracy |
|--------|----------|
| SimCLR + Prototype (Classifier1 approach) | 41.2% |
| SimCLR + Win-Rate (no pairwise training) | 35.3% |
| **Classifier2 (with pairwise training)** | **82.4%** |

**Conclusion**: Pairwise training approximately **doubles** single-image classification accuracy!

---

## Transfer Learning Approach

This system uses **transfer learning** rather than training a deep model from scratch:

| Component | Parameters | Training |
|-----------|------------|----------|
| ResNet-18 encoder | ~11M | Pretrained (SimCLR on unlabeled RHEED), fine-tuned |
| Reward head | ~130K | Trained from scratch with pairwise labels |

The encoder was pretrained using **SimCLR** (self-supervised contrastive learning) on unlabeled RHEED images. This allows us to train effectively with relatively few pairwise labels (~700 comparisons).

---

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 16 |
| Epochs | 30 |
| LR Schedule | Cosine Annealing |
| Gradient Clipping | max_norm = 1.0 |
| Samples per Epoch | 500 |

---

## Summary

| Component | Purpose |
|-----------|---------|
| **Pairwise labels** | Learn relative ranking within each dimension |
| **Ideal references (all 5 types)** | Anchor high scores + enable 5-class win-rate classification |
| **Bad images** | Anchor low scores + enable "Bad" detection |
| **Confidence weighting** | Weight reliable labels more heavily |
| **Bradley-Terry loss** | Convert comparisons to differentiable training |
| **Win-rate inference** | Calibrate dimensions via proper probability aggregation |
| **Cross-type comparison** | Classification (beat other types on my dimension?) |
| **Same-type comparison** | Quality score (percentile among my type's ideals) |
| **Bad detection** | Compare against bad refs to filter low-quality images |

---

*Document updated: January 2025 (v2.1)*
