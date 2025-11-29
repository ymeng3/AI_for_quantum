# Specific Simulator Augmentation Instructions for Undergraduate Student

## Simulator Overview

**Tool**: RHEED Simulator (Wolfram Mathematica)
- **Location**: `RHEED Simulator/` folder
- **Main file**: `RHEED Simulator.cdf` (self-contained) or `RHEED Simulator.nb` (source)
- **Requirements**: Wolfram Mathematica version >11
- **Documentation**: `Manual.pdf`, `Simulation_Parameters.xlsx`, `Supplementary Material.pdf`

## Key Parameters Identified

From the simulator code, these parameters can be adjusted:

1. **Intensity** - Pattern brightness/intensity
2. **Brightness** - Overall image brightness
3. **Angle** - Electron beam angle (incident angle)
4. **Lattice parameters** - Surface structure (default: 1×1 square lattice)
5. **Pattern display parameters** - How pattern is rendered

## Step-by-Step Implementation Plan

### Week 1: Simulator Setup & Exploration

**Day 1-2: Setup**
1. Install/access Wolfram Mathematica
2. Open `RHEED Simulator.cdf` or `RHEED Simulator.nb`
3. Read `Manual.pdf` to understand the interface
4. Generate a test pattern to understand the workflow

**Day 3-5: Parameter Exploration**
1. Open `Simulation_Parameters.xlsx` to see example parameters
2. Systematically test each adjustable parameter:
   - **Intensity**: Try different values (e.g., 0.5, 1.0, 1.5, 2.0)
   - **Brightness**: Try different values (e.g., 0.3, 0.5, 0.7, 1.0)
   - **Angle**: Try different incident angles
   - **Lattice parameters**: Try different surface structures
3. For each parameter combination:
   - Generate pattern
   - Compare visually with real RT13 seed images (in `data/STO_ideal_RT13/`)
   - Compare visually with real HTR seed images (in `data/STO_ideal_HTR/`)
   - Document which parameters create RT13-like patterns
   - Document which parameters create HTR-like patterns

**Deliverable**: 
- Parameter exploration notebook/document
- List of parameter ranges that produce RT13-like patterns
- List of parameter ranges that produce HTR-like patterns
- Notes on visual comparison with real images

### Week 2: Parameter Range Determination

**Goal**: Determine realistic parameter ranges for RT13 and HTR

**Tasks**:
1. **For RT13 patterns**:
   - Start with parameters that visually match real RT13 images
   - Systematically vary each parameter to find:
     - Minimum value that still looks like RT13
     - Maximum value that still looks like RT13
     - Typical/center value
   - Document the range: `[min, center, max]` for each parameter

2. **For HTR patterns**:
   - Start with parameters that visually match real HTR images
   - Systematically vary each parameter to find ranges
   - Document the ranges

3. **Create parameter sets**:
   - RT13 parameter set: `{intensity: [0.8, 1.2], brightness: [0.4, 0.6], angle: [...], ...}`
   - HTR parameter set: `{intensity: [0.9, 1.3], brightness: [0.5, 0.7], angle: [...], ...}`

**Deliverable**:
- Documented parameter ranges for RT13
- Documented parameter ranges for HTR
- Example parameter sets that produce realistic patterns

### Week 3-4: Automated Generation Pipeline

**Goal**: Create script to generate many synthetic images

**Approach Options**:

#### Option A: Mathematica Script (If student knows Mathematica)
- Create Mathematica script that:
  - Loops through parameter combinations
  - Generates patterns
  - Saves images
  - Exports to Python-readable format

#### Option B: Python + Mathematica API (Recommended)
- Use Mathematica's Python API or command-line interface
- Create Python script that:
  - Generates parameter combinations
  - Calls Mathematica to generate patterns
  - Processes and saves images

#### Option C: Manual Batch (If automation is difficult)
- Create parameter list in Excel/CSV
- Manually generate images in batches
- Organize and process images

**Generation Strategy**:

1. **Start Small**: Generate 50-100 images per class first
   - Test if they help training
   - Validate they look realistic
   - Check if they improve performance

2. **Parameter Sampling**:
   ```python
   # Pseudocode
   for class in ['RT13', 'HTR']:
       for i in range(num_images):
           # Sample from determined ranges
           params = {
               'intensity': random.uniform(rt13_intensity_min, rt13_intensity_max),
               'brightness': random.uniform(rt13_brightness_min, rt13_brightness_max),
               'angle': random.uniform(rt13_angle_min, rt13_angle_max),
               # ... other parameters
           }
           generate_and_save_pattern(params, f'SYNTH_{class}_{i:04d}.png')
   ```

3. **Image Processing**:
   - Convert to same format as real images (224×224, grayscale)
   - Apply same preprocessing as real images (if needed)
   - Save in organized folders: `data/Synthetic/RT13/` and `data/Synthetic/HTR/`

**Deliverable**:
- Generation script (Python or Mathematica)
- Initial synthetic dataset (50-100 per class for testing)
- Image processing pipeline

### Week 5-6: Integration & Training

**Goal**: Train models with synthetic + real data

**Tasks**:

1. **Organize Data**:
   ```
   data/
   ├── STO_ideal_RT13/     (24 real images)
   ├── STO_ideal_HTR/       (23 real images)
   └── Synthetic/
       ├── RT13/            (synthetic RT13 images)
       └── HTR/             (synthetic HTR images)
   ```

2. **Modify Training Code**:
   - Update data loading to include synthetic images
   - Implement mixing strategy:
     - **Option 1**: Mix in batches (e.g., 50% real, 50% synthetic)
     - **Option 2**: Pretrain on synthetic, fine-tune on real
     - **Option 3**: Weighted sampling (real data has higher weight)

3. **Run Experiments**:
   - **Baseline**: 47 real images only (current model)
   - **Exp 1**: 47 real + 100 synthetic per class
   - **Exp 2**: 47 real + 200 synthetic per class
   - **Exp 3**: 47 real + 500 synthetic per class

4. **Evaluate**:
   - Compare embedding separation ratios
   - Compare test performance (precision, recall, F1)
   - Check training accuracy (should decrease from 100% if overfitting is reduced)

**Deliverable**:
- Modified training pipeline
- Trained models for each experiment
- Performance comparison table

### Week 7-8: Optimization & Finalization

**Goal**: Optimize and finalize the approach

**Tasks**:
1. Analyze which experiment worked best
2. If results are promising:
   - Generate more synthetic images (scale up to 500-1000 per class)
   - Refine parameters based on results
   - Train final model
3. If results are not promising:
   - Adjust parameter ranges
   - Try different mixing strategies
   - Investigate why synthetic images aren't helping
4. Document final approach and results

**Deliverable**:
- Final optimized synthetic dataset
- Final trained model
- Comprehensive report with:
  - Parameter ranges used
  - Generation process
  - Training approach
  - Performance improvements
  - Lessons learned

## Specific Questions to Answer

1. **What parameters create RT13 patterns?**
   - Document parameter values that produce RT13-like patterns
   - Compare with real RT13 seed images

2. **What parameters create HTR patterns?**
   - Document parameter values that produce HTR-like patterns
   - Compare with real HTR seed images

3. **How many synthetic images do we need?**
   - Start with 100 per class
   - Scale up if helpful (200, 500, 1000)
   - Stop if more doesn't help

4. **What mixing strategy works best?**
   - Test different ratios (real:synthetic)
   - Test pretraining vs. mixed training
   - Choose best approach

5. **Do synthetic images improve performance?**
   - Compare metrics before/after
   - Check if overfitting is reduced
   - Validate on test set

## Expected Outcomes

### Success Criteria
- ✅ Synthetic images look realistic (visually similar to real patterns)
- ✅ Training accuracy decreases from 100% (less overfitting)
- ✅ Test performance improves (better precision/recall)
- ✅ Embedding separation ratio maintained or improved

### Potential Issues & Solutions

**Issue**: Synthetic images don't look realistic
- **Solution**: Adjust parameters, consult with me for guidance

**Issue**: Synthetic images don't improve performance
- **Solution**: Try different parameter ranges, different mixing strategies

**Issue**: Generation is too slow
- **Solution**: Generate in batches, optimize script, use parallel processing if possible

## Resources Provided

- Simulator files in `RHEED Simulator/` folder
- Real seed images: `data/STO_ideal_RT13/` (24 images) and `data/STO_ideal_HTR/` (23 images)
- Current training code: `src/classification.py`
- Pipeline documentation: `artifacts/pipeline_summary.md`
- Project repository: GitHub (you'll get access)

## Communication Plan

- **Weekly meetings**: Review progress, answer questions
- **As needed**: Help with Mathematica, parameter tuning, code debugging
- **Milestone check-ins**: End of each phase

## Final Notes

- **Start small**: Don't generate thousands of images immediately
- **Validate early**: Check if synthetic images look realistic before generating many
- **Iterate**: Adjust parameters based on results
- **Document**: Keep notes on what works and what doesn't
- **Ask questions**: Don't hesitate to ask if stuck or unsure

Good luck! This is a high-impact project that will significantly improve our classification model.

