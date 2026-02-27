# Classifier2: Unified Multi-Source Classification

## Overview

This classifier combines **three data sources** to train a robust RHEED image classifier:
1. **Pairwise comparison data** - Relative judgments from labeling software
2. **Bad image data** - Quality filtering from absolute scoring
3. **Ideal reference images** - High-quality examples for HTR and RT13

## Data Summary (as of 2025-12-14)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INVENTORY                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PAIRWISE COMPARISONS:        235 total                              │
│    - Unique image pairs:      90                                     │
│    - Unique images:           159                                    │
│    - Reconstruction types:    5 (+ 1 "Other")                        │
│    - Labelers:               SY (129), AJ (106)                     │
│                                                                      │
│  BAD IMAGES:                  13 images marked as unusable           │
│                                                                      │
│  IDEAL REFERENCE IMAGES:      40 total                               │
│    - STO_ideal_HTR:           19 images                              │
│    - STO_ideal_RT13:          21 images                              │
│                                                                      │
│  TRAJECTORY IMAGES (2022):    1,124 total                            │
│    - Currently labeled:       170 (15.1% coverage)                   │
│    - Unlabeled:               954 images                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Sources

### 1. Pairwise Comparison Data
- **Source**: Labeling Software (https://image-labeling-software.onrender.com)
- **File**: `Quantum Label Data - Pairwise_Comparison.csv`

| Column | Description |
|--------|-------------|
| Image1_Path | Path to first image |
| Image1_Name | Filename of first image |
| Image2_Path | Path to second image |
| Image2_Name | Filename of second image |
| Reconstruction_Type | One of: `(1 x 1)`, `Twinned(2 x 1)`, `c(6 x 2)`, `(√13 x √13)`, `HTR` |
| Winner | `1` (image1), `2` (image2), `tie`, or `not_apply` |
| Labeler_Name | Who made the comparison |

### 2. Bad Image Data
- **Source**: Absolute Scoring mode in labeling software
- **File**: `Quantum Label Data - Absolute_Scoring.csv`
- **Usage**: 6th class for quality filtering

### 3. Ideal Reference Images
- **Source**: `data/STO_ideal_HTR/` and `data/STO_ideal_RT13/`
- **Usage**: Strong positive examples for HTR and (√13 x √13) classes

## Approaches for Learning from Pairwise Data

### 1. Bradley-Terry Reward Model (Recommended for Starting)

The Bradley-Terry model is the standard approach used in RLHF (Reinforcement Learning from Human Feedback). It learns a scalar "reward" (or score) for each image such that the probability of preferring image A over B is:

```
P(A > B) = exp(r(A)) / (exp(r(A)) + exp(r(B))) = sigmoid(r(A) - r(B))
```

**Implementation:**
1. Use a shared encoder (e.g., ResNet18 from Classifier1) to embed both images
2. Add a reward head that outputs a scalar score per reconstruction type
3. Train with binary cross-entropy loss on preference pairs

**Loss function:**
```python
def bradley_terry_loss(r_chosen, r_rejected):
    # r_chosen: reward for preferred image
    # r_rejected: reward for non-preferred image
    return -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
```

### 2. Pairwise Preference Model (More Flexible)

Instead of learning scalar scores, directly model the preference probability:

```python
P(A > B | reconstruction_type) = model(concat(embed(A), embed(B)))
```

This can capture non-transitive preferences and context-dependent comparisons.

### 3. Contrastive Learning with Preference Margin

Extend SimCLR-style contrastive learning to use preference labels:
- Pull together embeddings of images with similar reconstruction type preferences
- Push apart embeddings where preferences differ

### 4. Siamese Network with Bradley-Terry Loss

Use a siamese architecture that:
1. Embeds both images with shared weights
2. Computes difference/interaction features
3. Predicts preference per reconstruction type

## Key Considerations

### Handling "Not Apply" Labels
When `winner = 'not_apply'`, neither image shows the reconstruction type. This provides negative signal:
- Both images should have low scores for that reconstruction type

### Handling "Tie" Labels
When `winner = 'tie'`, both images are equally representative:
- Their scores should be similar for that reconstruction type

### Multi-Reconstruction Types
Each image can have multiple reconstruction types. The model should output scores for all 5 types.

## References

1. [RLHF-Reward-Modeling Repository](https://github.com/RLHFlow/RLHF-Reward-Modeling) - State-of-the-art reward model training
2. [Bradley-Terry Model for Image Quality Assessment](https://www.researchgate.net/publication/2930678) - Pairwise comparison for images
3. [Fast Adaptation with Bradley-Terry Preference Models in Text-To-Image](https://arxiv.org/abs/2308.07929) - Adapting multimodal models with preferences
4. [Pairwise-RL Framework](https://arxiv.org/abs/2504.04950) - Unified pairwise approach for RLHF
5. [Rethinking Bradley-Terry Models](https://arxiv.org/html/2411.04991v1) - Foundations and alternatives

---

## Recommended Training Strategy

Given the data we have, here's the optimal approach:

### Strategy: Hybrid Bradley-Terry + Ideal Anchoring

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STAGE 1: Pretrain Encoder (Transfer from Classifier1)              │
│  ─────────────────────────────────────────────────────              │
│  • Use SimCLR encoder from Classifier1/artifacts/encoders/          │
│  • Already trained on RHEED domain                                  │
│                                                                      │
│  STAGE 2: Train Bradley-Terry Reward Model                          │
│  ─────────────────────────────────────────────────────              │
│  Loss components:                                                    │
│                                                                      │
│  L_pairwise: Bradley-Terry loss on human comparisons                │
│    - When winner=1: maximize P(img1 > img2)                         │
│    - When winner=2: maximize P(img2 > img1)                         │
│    - When tie: minimize |r(img1) - r(img2)|                         │
│    - When not_apply: push both scores low                           │
│                                                                      │
│  L_ideal: Anchor loss using ideal images                            │
│    - Ideal HTR images should have HIGH HTR scores                   │
│    - Ideal RT13 images should have HIGH (√13 x √13) scores          │
│    - Create synthetic pairs: ideal vs random trajectory             │
│                                                                      │
│  L_bad: Quality filtering loss                                      │
│    - Bad images should have LOW scores for all types                │
│    - Or: train separate "quality" head                              │
│                                                                      │
│  STAGE 3: Semi-Supervised Expansion (Optional)                      │
│  ─────────────────────────────────────────────────────              │
│  • Use trained model to pseudo-label unlabeled images               │
│  • Iteratively expand training set                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Approach?

1. **Pairwise data is efficient**: 235 comparisons provide relative ranking info for 159 images
2. **Ideal images provide anchors**: Without them, scores are only relative; ideal images set absolute scale
3. **Bad images define quality boundary**: Separates "which reconstruction" from "is it usable"
4. **Transfer learning**: Classifier1's encoder already understands RHEED patterns

### Data Augmentation Strategy

Given limited data, use heavy augmentation:
- **Pairwise**: Flip which image is "1" vs "2" (doubles data)
- **Ideal anchoring**: Generate many pairs (ideal + random)
- **Standard**: rotation, brightness, contrast, blur

### Output Format

The model outputs 6 scores per image:
```python
scores = model(image)  # [batch, 6]
# scores[:, 0] = (1 x 1) score
# scores[:, 1] = Twinned(2 x 1) score
# scores[:, 2] = c(6 x 2) score
# scores[:, 3] = (√13 x √13) score
# scores[:, 4] = HTR score
# scores[:, 5] = Quality score (high = good, low = bad)
```

---

## Files

- `analyze_data.py` - Comprehensive data analysis
- `fetch_pairwise_data.py` - Fetch data from Google Sheets
- `bradley_terry_model.py` - Bradley-Terry reward model implementation
- `train_unified.py` - Unified training with all data sources
- `classify_image.py` - Inference with trained model
