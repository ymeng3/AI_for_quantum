# NMF — Result Summarization

*Extracted from `Classifier2/PROGRESS.md` §2–2.4 (context: finding an RL reward signal for MBE growth control).*

The end goal: train an RL agent that controls MBE growth (temperature, oxygen pressure, deposition rate) to drive the surface toward a target reconstruction. The agent needs a **reward signal** at every step.

## 1. What we tried first: NMF on raw pixels

**Hypothesis.** RHEED frames are non-negative; if we run NMF on a stack of trajectory frames `V ≈ W·H`, the basis components `H` should correspond to physical reconstruction patterns, and the per-frame weights `W` should track phase transitions over time. This would give an unsupervised, low-dimensional, interpretable state representation — perfect for RL.

**What worked (within a single same-setup session):**
- Same-year decompositions converge cleanly (~7-10% reconstruction error)
- Same-setup HTR consistency ≈ 0.91-0.93
- Same-setup RT13 consistency ≈ 0.99
- Real temperature-correlated phase transitions are visible in W over time
- Frame-by-frame story matches physics: e.g. RR220411A at 539°C is dominated by component 1 (high-temp phase); at 741°C component 2 emerges and becomes the √13×√13 reconstruction; on cooldown component 2 strengthens.

NMF is a **solid real-time monitoring tool within a single experimental session** — that part works.

## 2. Why NMF failed for our actual goal

### Failure mode 1: Pixel-space NMF picks brightness, not diffraction

NMF minimizes total pixel reconstruction error. Given 150 ideal images across 4 classes and 4 components, NMF asks "what 4 templates minimize total pixel error?" — the answer is templates that capture **brightness gradients and screen geometry**, because those account for ~95-98% of pixel variance. The diffraction spot differences between RT13 and HTR are maybe **1-2% of total pixel variance**; NMF has no reason to spend a component on that when it can reduce more error by modeling the easy stuff.

Concrete result: **HTR vs RT13 cosine similarity in NMF basis space = 0.987** — they look essentially identical to the algorithm. C1, C3, C4 all became "HTR-like" and RT13 got no dedicated component. The basis components are physically meaningful (specular, diffraction spots, etc.) but **don't separate classes**.

This is a property of NMF's objective, not a tuning issue. More components and stronger regularization don't fix it — they just split the brightness explanation into more pieces.

### Failure mode 2: Cross-setup projection breaks

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

### Failure mode 3: Preprocessing helps RT13 but hurts HTR

Tried CLAHE, percentile normalization, and histogram equalization to align brightness distributions before projection.

| Method | RT13 cross-year sim | RT13 same-year sim | RT13 error | HTR cross-year sim | HTR same-year sim | HTR error |
|---|---|---|---|---|---|---|
| Raw | 0.248 | 0.997 | 12.9% | 0.576 | 0.909 | 15.9% |
| CLAHE | **0.586** | 0.991 | 7.1% | **0.195** | 0.927 | 8.2% |
| Percentile | 0.586 | 0.990 | 7.1% | 0.193 | 0.926 | 8.2% |
| HistEq | 0.524 | 0.958 | 5.7% | 0.203 | 0.496 | 7.2% |

CLAHE more than doubled RT13 cross-year consistency (0.248 → 0.586), but actually made HTR cross-year *worse* (0.576 → 0.195). After normalizing intensity, what's left is a **genuine structural difference** in the RHEED setup between 2022 and 2025 — a global pixel correction can't fix a different camera angle, phosphor screen state, or beam alignment. **The 2022 vs 2025 gap is a real physical setup change, not a data quality issue.**

### Failure mode 4: Ideal-image-trained NMF doesn't transfer either

Tried training NMF only on the 99 cleaned ideal images so the H basis would be "physically meaningful" by construction, then projecting trajectory frames. Same problem reappears: H is learned to reconstruct 2022-style ideal pixels, and a 2025 trajectory frame still looks nothing like any linear combination of those.

### Why W weights aren't temporally smooth

NMF treats each frame independently — no temporal regularization. So you get sudden jumps even when consecutive frames are nearly identical (NMF found two equally valid decompositions and picked different ones), and noisy flickering when a frame sits between two components. **Fix:** rolling-average post-processing on W if NMF is used at all. (Not a deep fix — just cosmetic for monitoring.)

## 3. Diagnosis

This is a **domain gap problem**, not an NMF objective problem. The 2025 frames and the 2022 ideal basis live in different pixel spaces. Even fixing NMF's objective (semi-supervised NMF à la Lee/Choi 2010, discriminative NMF à la Zafeiriou/Petrou 2010, graph-regularized variants) operates on **raw pixels** where brightness still dominates. They are incremental improvements on the same fundamental problem.

## 4. What should solve it

Two practical paths:

| Option | What it is | Status |
|---|---|---|
| Histogram matching → NMF | Match each trajectory frame's pixel distribution to the ideal images before projection | Tried (CLAHE) — partial success on RT13, fails on HTR |
| **Feature-space projection** | Extract features from Classifier2 (which was *trained* to be domain-invariant via cross-setup pairwise comparisons), then run NMF on those features instead of raw pixels | The principled fix |
| **Classifier directly** | Skip NMF entirely; use Classifier2 reward scores as the live signal | Recommended |
