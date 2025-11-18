# Simulator-Based Data Augmentation Guide

## Overview

This guide provides detailed instructions for using the RHEED Simulator to generate synthetic training data for improving the classification model. The simulator is a Wolfram Mathematica-based tool that generates theoretical RHEED patterns.

## Simulator Information

- **Tool**: RHEED Simulator (Wolfram Mathematica)
- **Files**: 
  - `RHEED Simulator.cdf` - Self-contained executable
  - `RHEED Simulator.nb` - Source code
  - `Simulation_Parameters.xlsx` - Parameter examples
  - `Manual.pdf` - User guide
- **Requirements**: Wolfram Mathematica (version >11)

## Understanding the Simulator

The simulator generates **theoretical RHEED patterns** based on:
- Lattice structure (1×1 square lattice by default)
- Electron beam parameters
- Surface structure parameters
- Pattern intensity and contrast

**Important**: These are theoretical patterns, so they may differ from experimental images. We need to ensure synthetic images are realistic enough to help the model.

---

## Data Augmentation Strategy

### Goal
Generate diverse synthetic RHEED patterns that:
1. Cover the range of RT13 and HTR patterns
2. Include realistic variations
3. Help reduce overfitting
4. Improve model generalization

### Target Dataset Size

**Recommended approach:**
- **Start small**: 200-500 synthetic images per class (RT13, HTR)
- **Scale up**: 1000-2000 images per class if initial results are promising
- **Total**: 400-4000 synthetic images (vs. 47 real images)

**Rationale:**
- Need enough diversity to help generalization
- But not so many that training becomes slow
- Can always generate more if needed

---

## Parameters to Adjust

Based on the simulator structure, here are key parameters to vary:

### 1. Lattice Structure Parameters
- **Lattice type**: Square, rectangular, etc.
- **Lattice constants**: a, b (spacing between atoms)
- **Surface orientation**: Different crystal orientations
- **Superstructure**: For RT13 vs. HTR patterns

### 2. Electron Beam Parameters
- **Beam energy**: Electron energy (typically 10-30 keV)
- **Beam angle**: Incident angle of electron beam
- **Azimuthal angle**: Rotation angle
- **Beam current**: Affects pattern intensity

### 3. Pattern Display Parameters
- **Intensity scaling**: Overall brightness
- **Contrast**: Pattern contrast
- **Noise level**: Add realistic noise
- **Background level**: Dark background intensity
- **Pattern resolution**: Image resolution

### 4. Surface Structure Parameters
- **Surface reconstruction**: Different surface terminations
- **Step density**: Surface roughness
- **Domain size**: Size of ordered regions
- **Defect density**: Point defects, vacancies

### 5. RT13-Specific Parameters
- **13×13 reconstruction parameters**
- **Domain boundaries**
- **Phase transitions**

### 6. HTR-Specific Parameters
- **High-temperature reconstruction parameters**
- **Thermal fluctuations**
- **Disorder parameters**

---

## Step-by-Step Implementation Plan

### Phase 1: Understanding the Simulator 

**Tasks:**
1. Install Wolfram Mathematica (if not available)
2. Open `RHEED Simulator.cdf` or `.nb` file
3. Read `Manual.pdf` to understand interface
4. Review `Simulation_Parameters.xlsx` for example parameters
5. Generate a few test patterns manually
6. Compare synthetic patterns with real RT13/HTR seed images

**Deliverables:**
- Understanding of simulator interface
- List of adjustable parameters
- Notes on how synthetic patterns compare to real ones

### Phase 2: Parameter Exploration 

**Tasks:**
1. Identify which parameters create RT13-like patterns
2. Identify which parameters create HTR-like patterns
3. Determine realistic parameter ranges:
   - Minimum/maximum values for each parameter
   - Which parameters have most impact
   - Which parameters create realistic variations
4. Create parameter sets for RT13 and HTR

**Method:**
- Start with parameters from `Simulation_Parameters.xlsx`
- Systematically vary one parameter at a time
- Visually compare with real seed images
- Document parameter ranges that produce realistic patterns

**Deliverables:**
- Parameter ranges for RT13 patterns
- Parameter ranges for HTR patterns
- Documentation of parameter effects

### Phase 3: Automated Generation 

**Tasks:**
1. Create script to automate parameter variation
2. Generate diverse parameter combinations:
   - Use grid search or random sampling
   - Ensure coverage of parameter space
   - Avoid unrealistic combinations
3. Generate images:
   - 200-500 images per class initially
   - Save with consistent naming: `SYNTH_RT13_0001.png`, `SYNTH_HTR_0001.png`
   - Save in same format as real images (224×224, grayscale)
4. Validate generated images:
   - Visual inspection
   - Compare statistics with real images
   - Check for artifacts or unrealistic patterns

**Script Structure:**
```python
# Pseudocode
for class in ['RT13', 'HTR']:
    for i in range(num_images):
        # Sample parameters from defined ranges
        params = sample_parameters(class)
        
        # Call Mathematica simulator (via API or script)
        image = generate_pattern(params)
        
        # Preprocess to match real image format
        image = preprocess_image(image)
        
        # Save
        save_image(image, f'SYNTH_{class}_{i:04d}.png')
```

**Deliverables:**
- Automated generation script
- Initial synthetic dataset (200-500 per class)
- Validation report

---

## Parameter Variation Strategy

### Systematic Approach

**Grid Search (for key parameters):**
```python
# Example for 2 key parameters
for param1 in [val1_min, val1_mid, val1_max]:
    for param2 in [val2_min, val2_mid, val2_max]:
        generate_pattern(param1, param2)
```

**Random Sampling (for many parameters):**
```python
# Sample from parameter distributions
for i in range(num_images):
    params = {
        'energy': random.uniform(10, 30),  # keV
        'angle': random.uniform(0, 5),     # degrees
        'intensity': random.uniform(0.5, 1.5),
        # ... other parameters
    }
    generate_pattern(**params)
```

**Latin Hypercube Sampling (for better coverage):**
- Ensures good coverage of parameter space
- More efficient than random sampling
- Use if you have many parameters

### Parameter Ranges (To Be Determined)

**You need to determine these by experimentation:**

1. **RT13 Parameters:**
   - Lattice constants: [range to be determined]
   - Beam energy: [range to be determined]
   - Pattern intensity: [range to be determined]
   - ... (other parameters)

2. **HTR Parameters:**
   - Lattice constants: [range to be determined]
   - Beam energy: [range to be determined]
   - Pattern intensity: [range to be determined]
   - ... (other parameters)

**How to determine:**
- Start with values from `Simulation_Parameters.xlsx`
- Compare synthetic patterns with real seed images
- Adjust until synthetic patterns look realistic
- Document the ranges

---

## Quality Control

### Visual Inspection
- Compare synthetic images with real seed images
- Check for:
  - Realistic pattern structure
  - Appropriate contrast
  - Realistic noise levels
  - No obvious artifacts

### Statistical Comparison
- Compare image statistics:
  - Mean intensity
  - Standard deviation
  - Histogram distribution
- Synthetic should be similar to real (but can have variation)

### Embedding Space Validation
- Generate embeddings for synthetic images
- Check if they cluster with real images of same class
- If synthetic images are in wrong cluster → adjust parameters

### Training Validation
- Train model with synthetic data
- Check if performance improves
- If performance degrades → synthetic images may be unrealistic

---

## Expected Challenges & Solutions

### Challenge 1: Simulator-Real Gap
**Problem**: Synthetic images may look different from real images
**Solution**: 
- Adjust parameters to match real images
- Use domain adaptation techniques
- Weight real data more heavily in training

### Challenge 2: Parameter Space Too Large
**Problem**: Too many parameters to explore
**Solution**:
- Focus on most important parameters first
- Use parameter sensitivity analysis
- Start with parameters from examples

### Challenge 3: Generation Speed
**Problem**: Generating many images is slow
**Solution**:
- Batch generation
- Parallel processing if possible
- Generate in stages (start small, scale up)

### Challenge 4: Realism Validation
**Problem**: Hard to know if synthetic images are realistic
**Solution**:
- Expert review of sample images
- Embedding space validation
- Training performance validation

---

## Success Metrics

### Quantitative Metrics
1. **Embedding Separation Ratio**: Should improve or maintain (currently 23.63x)
2. **Test Performance**: Should improve precision/recall
3. **Overfitting Reduction**: Training accuracy should decrease (from 100%)
4. **Generalization**: Better performance on diverse test images

### Qualitative Metrics
1. **Visual Realism**: Synthetic images look like real patterns
2. **Diversity**: Synthetic images cover range of real patterns
3. **Class Separation**: RT13 and HTR synthetic images are distinct


---

## Resources

- Simulator files in `RHEED Simulator/` folder
- Real seed images in `data/STO_ideal_RT13/` and `data/STO_ideal_HTR/`
- Current training code in `src/classification.py`
- Pipeline documentation in `artifacts/pipeline_summary.md`

