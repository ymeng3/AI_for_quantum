"""
Setup Comparison: 2022 vs 2025 Trajectory Data

Analyzes how the two experimental setups differ:
  - Image dimensions and resolution
  - Pixel intensity distributions (brightness, contrast)
  - Mean and variance images per setup
  - Spatial structure (where signal lives)
  - Cosine similarity of mean images across setups
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent / "setup_comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAJ_ROOT = Path(__file__).parent.parent / "data" / "trajectories"

TRAJECTORIES = {
    '2022': [
        TRAJ_ROOT / '2022-02-04',
        TRAJ_ROOT / '2022-02-06',
        TRAJ_ROOT / '2022-04-11',
    ],
    '2025': [
        TRAJ_ROOT / '2025-10-04' / 'A',
        TRAJ_ROOT / '2025-10-04' / 'B',
        TRAJ_ROOT / '2025-10-05',
    ],
}

TARGET_SIZE = (320, 240)  # standardize for comparison


def load_images(directory, max_frames=200):
    """Load up to max_frames images from a directory, return list of grayscale arrays."""
    files = sorted(list(directory.glob('*.bmp')) + list(directory.glob('*.png')))
    if not files:
        return [], []
    step = max(1, len(files) // max_frames)
    files = files[::step][:max_frames]
    arrays = []
    raw_sizes = []
    for f in files:
        try:
            img = Image.open(f).convert('L')
            raw_sizes.append(img.size)
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            arrays.append(np.array(img, dtype=np.float32))
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    return arrays, raw_sizes


def compute_stats(arrays):
    """Compute per-pixel and global statistics for a set of images."""
    stack = np.stack(arrays, axis=0)  # (N, H, W)
    flat = stack.reshape(len(arrays), -1)  # (N, pixels)
    return {
        'mean_image': stack.mean(axis=0),
        'std_image': stack.std(axis=0),
        'global_mean': flat.mean(),
        'global_std': flat.std(),
        'global_median': np.median(flat),
        'p5': np.percentile(flat, 5),
        'p25': np.percentile(flat, 25),
        'p75': np.percentile(flat, 75),
        'p95': np.percentile(flat, 95),
        'max': flat.max(),
        'min': flat.min(),
        'all_pixels': flat.flatten(),
        'per_frame_mean': flat.mean(axis=1),
        'per_frame_std': flat.std(axis=1),
        'per_frame_max': flat.max(axis=1),
    }


# ── Load all data ──────────────────────────────────────────────────────────────
print("Loading images...")
setup_data = {}
raw_sizes_all = {}

for setup, dirs in TRAJECTORIES.items():
    all_arrays = []
    all_sizes = []
    for d in dirs:
        arrays, sizes = load_images(d)
        print(f"  {setup} {d.name}: {len(arrays)} frames, raw size = {sizes[0] if sizes else 'N/A'}")
        all_arrays.extend(arrays)
        all_sizes.extend(sizes)
    setup_data[setup] = all_arrays
    raw_sizes_all[setup] = all_sizes

print(f"\n2022 total frames loaded: {len(setup_data['2022'])}")
print(f"2025 total frames loaded: {len(setup_data['2025'])}")

stats_2022 = compute_stats(setup_data['2022'])
stats_2025 = compute_stats(setup_data['2025'])

# ── Print numerical report ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("NUMERICAL COMPARISON")
print("="*60)
print(f"{'Metric':<25} {'2022':>12} {'2025':>12}")
print("-"*50)

rows = [
    ("Raw image size",      f"{raw_sizes_all['2022'][0]}",  f"{raw_sizes_all['2025'][0]}"),
    ("N frames analyzed",   f"{len(setup_data['2022'])}",   f"{len(setup_data['2025'])}"),
    ("Mean pixel value",    f"{stats_2022['global_mean']:.1f}",  f"{stats_2025['global_mean']:.1f}"),
    ("Median pixel value",  f"{stats_2022['global_median']:.1f}",f"{stats_2025['global_median']:.1f}"),
    ("Std pixel value",     f"{stats_2022['global_std']:.1f}",   f"{stats_2025['global_std']:.1f}"),
    ("5th percentile",      f"{stats_2022['p5']:.1f}",     f"{stats_2025['p5']:.1f}"),
    ("25th percentile",     f"{stats_2022['p25']:.1f}",    f"{stats_2025['p25']:.1f}"),
    ("75th percentile",     f"{stats_2022['p75']:.1f}",    f"{stats_2025['p75']:.1f}"),
    ("95th percentile",     f"{stats_2022['p95']:.1f}",    f"{stats_2025['p95']:.1f}"),
    ("Max pixel value",     f"{stats_2022['max']:.1f}",    f"{stats_2025['max']:.1f}"),
]
for name, v22, v25 in rows:
    print(f"{name:<25} {v22:>12} {v25:>12}")

# KS test on pixel distributions
ks_stat, ks_p = stats.ks_2samp(
    np.random.choice(stats_2022['all_pixels'], 50000),
    np.random.choice(stats_2025['all_pixels'], 50000)
)
print(f"\nKS test (pixel distributions): stat={ks_stat:.4f}, p={ks_p:.2e}")
print("  (stat near 1 = very different distributions, near 0 = similar)")

# Cosine similarity of mean images
m22 = stats_2022['mean_image'].flatten()
m25 = stats_2025['mean_image'].flatten()
cos_sim = np.dot(m22, m25) / (np.linalg.norm(m22) * np.linalg.norm(m25))
print(f"\nCosine similarity of mean images: {cos_sim:.4f}")
print("  (1.0 = identical structure, 0 = no overlap)")

# Per-frame mean comparison
print(f"\nPer-frame mean brightness:")
print(f"  2022: {stats_2022['per_frame_mean'].mean():.1f} ± {stats_2022['per_frame_mean'].std():.1f}")
print(f"  2025: {stats_2025['per_frame_mean'].mean():.1f} ± {stats_2025['per_frame_mean'].std():.1f}")
print(f"  Ratio (2025/2022): {stats_2025['per_frame_mean'].mean() / stats_2022['per_frame_mean'].mean():.2f}x")

print(f"\nPer-frame contrast (std):")
print(f"  2022: {stats_2022['per_frame_std'].mean():.1f} ± {stats_2022['per_frame_std'].std():.1f}")
print(f"  2025: {stats_2025['per_frame_std'].mean():.1f} ± {stats_2025['per_frame_std'].std():.1f}")


# ── Figure 1: Pixel intensity histograms ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Full distribution
ax = axes[0]
ax.hist(np.random.choice(stats_2022['all_pixels'], 100000), bins=128,
        alpha=0.6, color='steelblue', label='2022', density=True)
ax.hist(np.random.choice(stats_2025['all_pixels'], 100000), bins=128,
        alpha=0.6, color='tomato', label='2025', density=True)
ax.set_xlabel('Pixel value (0–255)')
ax.set_ylabel('Density')
ax.set_title('Pixel Intensity Distribution')
ax.legend()

# Per-frame mean brightness
ax = axes[1]
ax.hist(stats_2022['per_frame_mean'], bins=40, alpha=0.7, color='steelblue', label='2022', density=True)
ax.hist(stats_2025['per_frame_mean'], bins=40, alpha=0.7, color='tomato', label='2025', density=True)
ax.set_xlabel('Mean pixel value per frame')
ax.set_title('Per-Frame Brightness Distribution')
ax.legend()

# Per-frame contrast (std)
ax = axes[2]
ax.hist(stats_2022['per_frame_std'], bins=40, alpha=0.7, color='steelblue', label='2022', density=True)
ax.hist(stats_2025['per_frame_std'], bins=40, alpha=0.7, color='tomato', label='2025', density=True)
ax.set_xlabel('Pixel std per frame (contrast)')
ax.set_title('Per-Frame Contrast Distribution')
ax.legend()

fig.suptitle('2022 vs 2025 Setup: Intensity Statistics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_intensity_distributions.png', dpi=150)
plt.close()
print("\nSaved fig1_intensity_distributions.png")


# ── Figure 2: Mean images and variance maps ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Mean images
im = axes[0, 0].imshow(stats_2022['mean_image'], cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('2022 Mean Image')
plt.colorbar(im, ax=axes[0, 0])

im = axes[0, 1].imshow(stats_2025['mean_image'], cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('2025 Mean Image')
plt.colorbar(im, ax=axes[0, 1])

# Difference of mean images
diff = stats_2022['mean_image'] - stats_2025['mean_image']
im = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-128, vmax=128)
axes[0, 2].set_title('Mean Image Difference (2022 − 2025)')
plt.colorbar(im, ax=axes[0, 2])

# Variance maps (std across frames)
vmax_std = max(stats_2022['std_image'].max(), stats_2025['std_image'].max())

im = axes[1, 0].imshow(stats_2022['std_image'], cmap='hot', vmin=0, vmax=vmax_std)
axes[1, 0].set_title('2022 Pixel Std (variance map)')
plt.colorbar(im, ax=axes[1, 0])

im = axes[1, 1].imshow(stats_2025['std_image'], cmap='hot', vmin=0, vmax=vmax_std)
axes[1, 1].set_title('2025 Pixel Std (variance map)')
plt.colorbar(im, ax=axes[1, 1])

# Ratio of std maps
ratio = (stats_2025['std_image'] + 1e-6) / (stats_2022['std_image'] + 1e-6)
im = axes[1, 2].imshow(ratio, cmap='coolwarm', vmin=0, vmax=4)
axes[1, 2].set_title('Std Ratio (2025 / 2022)\n>1 = more variance in 2025')
plt.colorbar(im, ax=axes[1, 2])

fig.suptitle('2022 vs 2025: Spatial Structure', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_mean_variance_maps.png', dpi=150)
plt.close()
print("Saved fig2_mean_variance_maps.png")


# ── Figure 3: Sample frames side by side ─────────────────────────────────────
n_samples = 6
idx22 = np.linspace(0, len(setup_data['2022'])-1, n_samples, dtype=int)
idx25 = np.linspace(0, len(setup_data['2025'])-1, n_samples, dtype=int)

fig, axes = plt.subplots(2, n_samples, figsize=(18, 7))

for i, idx in enumerate(idx22):
    axes[0, i].imshow(setup_data['2022'][idx], cmap='gray', vmin=0, vmax=255)
    axes[0, i].axis('off')
    axes[0, i].set_title(f'2022 #{idx}', fontsize=8)

for i, idx in enumerate(idx25):
    axes[1, i].imshow(setup_data['2025'][idx], cmap='gray', vmin=0, vmax=255)
    axes[1, i].axis('off')
    axes[1, i].set_title(f'2025 #{idx}', fontsize=8)

axes[0, 0].set_ylabel('2022', fontsize=12, rotation=90, labelpad=10)
axes[1, 0].set_ylabel('2025', fontsize=12, rotation=90, labelpad=10)

fig.suptitle('Sample Frames: 2022 vs 2025', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_sample_frames.png', dpi=150)
plt.close()
print("Saved fig3_sample_frames.png")


# ── Figure 4: Histogram-matched 2025 vs 2022 ─────────────────────────────────
# Show what 2025 looks like after matching to 2022's intensity distribution
sample_2022 = setup_data['2022'][len(setup_data['2022'])//2]
sample_2025 = setup_data['2025'][len(setup_data['2025'])//2]

# Manual histogram matching using CDF mapping
def match_histograms(source, reference):
    src_flat = source.flatten()
    ref_flat = reference.flatten()
    src_cdf = np.cumsum(np.bincount(src_flat.astype(np.uint8), minlength=256)) / src_flat.size
    ref_cdf = np.cumsum(np.bincount(ref_flat.astype(np.uint8), minlength=256)) / ref_flat.size
    lut = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        lut[v] = np.searchsorted(ref_cdf, src_cdf[v])
    return lut[source.astype(np.uint8)].astype(np.float32)

matched = match_histograms(sample_2025, sample_2022)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(sample_2022, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('2022 reference frame')
axes[0].axis('off')

axes[1].imshow(sample_2025, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('2025 original frame')
axes[1].axis('off')

axes[2].imshow(matched, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('2025 → histogram matched to 2022')
axes[2].axis('off')

axes[3].imshow(sample_2025 - matched, cmap='RdBu_r', vmin=-60, vmax=60)
axes[3].set_title('Difference (original − matched)')
axes[3].axis('off')

fig.suptitle('Effect of Histogram Matching (domain gap reduction)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_histogram_matching.png', dpi=150)
plt.close()
print("Saved fig4_histogram_matching.png")

print(f"\nAll results saved to: {OUTPUT_DIR}")
