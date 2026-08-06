"""
NMF with Ideal Image Basis — Downsampled Analysis

Step 3: Run NMF on ideal images (downsampled to grayscale 64x48),
verify component-class mapping, project all 6 trajectories onto
the ideal basis, compare HTR vs RT13, and report consistency findings.

Key improvements over v1:
  - Grayscale (not RGB) → removes 3x redundancy
  - Aggressive downsampling (64x48 = 3072 features vs 921,600)
  - Background suppressed, diffraction features emphasized
  - Detailed numerical metrics (confusion, cosine sim, silhouette)
  - Comprehensive text report
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
from PIL import Image, ImageDraw
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
IDEAL_DIR = Path(__file__).parent / "ideal_images"
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"
OUTPUT_DIR = Path(__file__).parent / "nmf_ideal_downsampled_results"

# Downsampled target size (grayscale)
DS_SIZE = (64, 48)  # width, height → 3072 features

# Specular mask scaled to DS_SIZE (original ~[280,200,310,250] at 640x480)
MASK_RECT_DS = [28, 20, 31, 25]

# Reconstruction classes
CLASSES = ['STO_ideal_1x1', 'STO_ideal_RT13', 'STO_ideal_HTR', 'STO_ideal_c6x2']
CLASS_LABELS = ['1x1', 'RT13', 'HTR', 'c6x2']
CLASS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Trajectories
TRAJECTORIES = {
    'HTR': [
        ('2025-10-04/A', 'HL251004A'),
        ('2025-10-04/B', 'HL251004B'),
        ('2022-02-06',   'RR220206A'),
    ],
    'RT13': [
        ('2025-10-05',  'HL251005A'),
        ('2022-02-04',  'RR220204A'),
        ('2022-04-11',  'RR220411A'),
    ]
}


# ─── Preprocessing ────────────────────────────────────────────────────────────

def load_and_preprocess(img_path):
    """Load image → grayscale → downsample → mask specular → normalize."""
    img = Image.open(img_path).convert('L')  # grayscale
    img = img.resize(DS_SIZE, Image.Resampling.LANCZOS)

    # Mask specular reflection
    draw = ImageDraw.Draw(img)
    draw.rectangle(MASK_RECT_DS, fill=0)

    arr = np.array(img, dtype=np.float64) / 255.0
    return arr


def load_ideal_images():
    """Load all ideal images, return (data_matrix, labels, per_class_counts)."""
    all_arrays = []
    labels = []
    counts = {}

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = IDEAL_DIR / cls_name
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} not found, skipping")
            counts[cls_name] = 0
            continue

        files = sorted(list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.bmp')))
        counts[cls_name] = len(files)

        for f in files:
            try:
                arr = load_and_preprocess(f)
                all_arrays.append(arr.flatten())
                labels.append(cls_idx)
            except Exception as e:
                print(f"  Error loading {f.name}: {e}")

    V = np.array(all_arrays)
    labels = np.array(labels)
    return V, labels, counts


def load_trajectory(traj_path):
    """Load all images from a trajectory, return data matrix."""
    traj_dir = TRAJECTORY_DIR / traj_path
    if not traj_dir.exists():
        print(f"  WARNING: {traj_dir} not found")
        return None, []

    files = sorted(list(traj_dir.glob('*.bmp')) + list(traj_dir.glob('*.png')))
    arrays = []
    for f in files:
        try:
            arr = load_and_preprocess(f)
            arrays.append(arr.flatten())
        except:
            pass

    if not arrays:
        return None, files
    return np.array(arrays), files


# ─── Analysis ─────────────────────────────────────────────────────────────────

def train_nmf(V, n_components=4, max_iter=1000):
    """Train NMF on data matrix V."""
    nmf = NMF(
        n_components=n_components,
        init='nndsvda',
        max_iter=max_iter,
        random_state=42,
    )
    W = nmf.fit_transform(V)
    H = nmf.components_
    return nmf, W, H


def compute_class_means(W, labels, n_classes):
    """Mean weight vector per class."""
    means = np.zeros((n_classes, W.shape[1]))
    stds = np.zeros((n_classes, W.shape[1]))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            means[c] = W[mask].mean(axis=0)
            stds[c] = W[mask].std(axis=0)
    return means, stds


def component_class_mapping(means):
    """Determine which component best represents each class and vice versa."""
    n_classes, n_comp = means.shape

    # Normalize means row-wise
    row_sums = means.sum(axis=1, keepdims=True) + 1e-10
    normed = means / row_sums

    # For each class, which component is dominant?
    class_to_comp = {}
    for c in range(n_classes):
        dom_comp = np.argmax(normed[c])
        class_to_comp[CLASS_LABELS[c]] = (dom_comp, normed[c, dom_comp])

    # For each component, which class activates it most?
    col_sums = means.sum(axis=0, keepdims=True) + 1e-10
    col_normed = means / col_sums
    comp_to_class = {}
    for k in range(n_comp):
        dom_class = np.argmax(col_normed[:, k])
        comp_to_class[k] = (CLASS_LABELS[dom_class], col_normed[dom_class, k])

    return class_to_comp, comp_to_class, normed


def classification_accuracy(W, labels):
    """Classify each ideal image by its dominant component; report accuracy."""
    preds = np.argmax(W, axis=1)

    # Build mapping: for each true class, find the most common component
    class_comp_map = {}
    for c in range(len(CLASS_LABELS)):
        mask = labels == c
        if mask.sum() > 0:
            counts = Counter(preds[mask])
            class_comp_map[c] = counts.most_common(1)[0][0]

    comp_assignments = list(class_comp_map.values())
    unique_assignments = len(set(comp_assignments))

    correct = 0
    total = len(labels)
    for i, true_class in enumerate(labels):
        if preds[i] == class_comp_map.get(true_class, -1):
            correct += 1

    return correct / total, class_comp_map, unique_assignments


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_basis_components(H, comp_to_class, output_path):
    """Visualize NMF basis components as grayscale images."""
    n_comp = H.shape[0]
    fig, axes = plt.subplots(1, n_comp, figsize=(4 * n_comp, 4))
    if n_comp == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        component = H[i].reshape(DS_SIZE[1], DS_SIZE[0])
        component = component / (component.max() + 1e-10)
        ax.imshow(component, cmap='hot', interpolation='nearest')
        cls_label, score = comp_to_class.get(i, ('?', 0))
        ax.set_title(f'C{i+1} → {cls_label} ({score:.0%})', fontsize=13)
        ax.axis('off')

    plt.suptitle('NMF Basis Components (downsampled grayscale, trained on ideals)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_ideal_decomposition(means, stds, normed, output_path):
    """Bar chart: mean component weights per class, with error bars."""
    n_classes, n_comp = means.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(n_classes)
    width = 0.8 / n_comp
    for k in range(n_comp):
        ax1.bar(x + k * width, means[:, k], width, yerr=stds[:, k],
                label=f'C{k+1}', capsize=3, alpha=0.85)
    ax1.set_xticks(x + width * (n_comp - 1) / 2)
    ax1.set_xticklabels(CLASS_LABELS)
    ax1.set_ylabel('Mean NMF Weight')
    ax1.set_title('Raw Component Weights by Class')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    for k in range(n_comp):
        ax2.bar(x + k * width, normed[:, k], width,
                label=f'C{k+1}', alpha=0.85)
    ax2.set_xticks(x + width * (n_comp - 1) / 2)
    ax2.set_xticklabels(CLASS_LABELS)
    ax2.set_ylabel('Normalized Weight (fraction)')
    ax2.set_title('Normalized Component Weights by Class')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Ideal Image Decomposition — Component-Class Mapping', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_confusion_matrix(W, labels, class_comp_map, output_path):
    """Confusion-style matrix: for each class, distribution over components."""
    n_classes = len(CLASS_LABELS)
    n_comp = W.shape[1]

    preds = np.argmax(W, axis=1)
    conf = np.zeros((n_classes, n_comp))
    for i in range(len(labels)):
        conf[labels[i], preds[i]] += 1

    row_totals = conf.sum(axis=1, keepdims=True) + 1e-10
    conf_norm = conf / row_totals

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction of images')

    for i in range(n_classes):
        for j in range(n_comp):
            count = int(conf[i, j])
            frac = conf_norm[i, j]
            color = 'white' if frac > 0.5 else 'black'
            ax.text(j, i, f'{count}\n({frac:.0%})', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    ax.set_xticks(range(n_comp))
    ax.set_xticklabels([f'C{k+1}' for k in range(n_comp)])
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel('Dominant NMF Component')
    ax.set_ylabel('True Reconstruction Class')
    ax.set_title('Component Assignment Confusion Matrix\n(count and fraction per class)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_cosine_similarity(W, labels, output_path):
    """Cosine similarity between class centroids in NMF space."""
    n_classes = len(CLASS_LABELS)
    centroids = np.zeros((n_classes, W.shape[1]))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = W[mask].mean(axis=0)

    sim = cosine_similarity(centroids)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if sim[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{sim[i,j]:.3f}', ha='center', va='center',
                    fontsize=12, color=color, fontweight='bold')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_title('Cosine Similarity Between Class Centroids\n(in NMF weight space)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_trajectory_decomposition(traj_data, comp_to_class, output_path):
    """6-panel: normalized component evolution for all trajectories."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    colors = CLASS_COLORS

    for row, (traj_type, traj_list) in enumerate(TRAJECTORIES.items()):
        for col, (traj_path, traj_name) in enumerate(traj_list):
            ax = axes[row, col]
            key = f"{traj_path.replace('/', '_')}_{traj_name}"

            if key not in traj_data:
                ax.text(0.5, 0.5, f'No data: {traj_name}',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            W = traj_data[key]['W']
            n_comp = W.shape[1]
            W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)

            for k in range(n_comp):
                cls_label = comp_to_class.get(k, ('?', 0))[0]
                ax.plot(W_norm[:, k], color=colors[k % len(colors)],
                        label=f'C{k+1}: {cls_label}', linewidth=1.2, alpha=0.85)

            ax.set_title(f'{traj_name} ({traj_type})', fontsize=13, fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Normalized Weight')
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Trajectory Decomposition — Ideal-Image NMF Basis (Downsampled)\n'
                 'Top: HTR trajectories | Bottom: RT13 trajectories', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_stacked_trajectories(traj_data, comp_to_class, output_path):
    """Stacked area plot for all 6 trajectories."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    colors = CLASS_COLORS

    for row, (traj_type, traj_list) in enumerate(TRAJECTORIES.items()):
        for col, (traj_path, traj_name) in enumerate(traj_list):
            ax = axes[row, col]
            key = f"{traj_path.replace('/', '_')}_{traj_name}"

            if key not in traj_data:
                continue

            W = traj_data[key]['W']
            n_comp = W.shape[1]
            W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)

            labels_stk = [f'C{k+1}: {comp_to_class.get(k, ("?",0))[0]}'
                          for k in range(n_comp)]
            ax.stackplot(range(len(W_norm)),
                         [W_norm[:, k] for k in range(n_comp)],
                         labels=labels_stk, colors=colors[:n_comp], alpha=0.75)

            ax.set_title(f'{traj_name} ({traj_type})', fontsize=13, fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Composition')
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Reconstruction Composition (Stacked) — Downsampled Ideal Basis\n'
                 'Top: HTR trajectories | Bottom: RT13 trajectories', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_htr_vs_rt13_summary(traj_data, comp_to_class, output_path):
    """Direct comparison: average composition of HTR vs RT13 trajectories."""
    n_comp = list(traj_data.values())[0]['W'].shape[1]

    avg_comp = {'HTR': [], 'RT13': []}
    for key, data in traj_data.items():
        W = data['W']
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)
        avg = W_norm.mean(axis=0)
        avg_comp[data['type']].append((data['name'], avg))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Per-trajectory average composition (stacked bar)
    ax = axes[0]
    all_names = []
    all_avgs = []
    for ttype in ['HTR', 'RT13']:
        for name, avg in avg_comp[ttype]:
            all_names.append(name)
            all_avgs.append(avg)

    x = np.arange(len(all_names))
    bottom = np.zeros(len(all_names))
    for k in range(n_comp):
        vals = [a[k] for a in all_avgs]
        cls_label = comp_to_class.get(k, ('?', 0))[0]
        ax.bar(x, vals, bottom=bottom, label=f'C{k+1}: {cls_label}',
               color=CLASS_COLORS[k % len(CLASS_COLORS)], alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=30, ha='right')
    ax.set_ylabel('Average Composition')
    ax.set_title('Average Composition per Trajectory')
    ax.legend(fontsize=8)
    ax.axvline(2.5, color='black', linestyle='--', alpha=0.5)
    ax.text(1, 1.02, 'HTR', ha='center', transform=ax.get_xaxis_transform(), fontweight='bold')
    ax.text(4, 1.02, 'RT13', ha='center', transform=ax.get_xaxis_transform(), fontweight='bold')

    # Panel 2: HTR vs RT13 group means
    ax = axes[1]
    htr_mean = np.mean([a for _, a in avg_comp['HTR']], axis=0)
    rt13_mean = np.mean([a for _, a in avg_comp['RT13']], axis=0)

    x2 = np.arange(n_comp)
    width = 0.35
    for k in range(n_comp):
        ax.bar(x2[k] - width/2, htr_mean[k], width, color='coral',
               label='HTR avg' if k == 0 else '')
        ax.bar(x2[k] + width/2, rt13_mean[k], width, color='steelblue',
               label='RT13 avg' if k == 0 else '')

    ax.set_xticks(x2)
    ax.set_xticklabels([f'C{k+1}: {comp_to_class.get(k, ("?",0))[0]}' for k in range(n_comp)])
    ax.set_ylabel('Mean Normalized Weight')
    ax.set_title('HTR vs RT13: Mean Component Weights')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Difference (HTR - RT13)
    ax = axes[2]
    diff = htr_mean - rt13_mean
    bar_colors = ['coral' if d > 0 else 'steelblue' for d in diff]
    ax.bar(x2, diff, color=bar_colors, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels([f'C{k+1}: {comp_to_class.get(k, ("?",0))[0]}' for k in range(n_comp)])
    ax.set_ylabel('HTR mean − RT13 mean')
    ax.set_title('Component Weight Difference (HTR − RT13)')
    ax.grid(axis='y', alpha=0.3)
    for i, d in enumerate(diff):
        ax.text(i, d + 0.005 * np.sign(d), f'{d:+.3f}', ha='center', fontsize=10)

    plt.suptitle('HTR vs RT13 Trajectory Comparison Summary', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# ─── Text Report ──────────────────────────────────────────────────────────────

def generate_report(
    counts, nmf, V_ideal, W_ideal, labels, means, stds, normed,
    class_to_comp, comp_to_class, accuracy, class_comp_map, n_unique,
    traj_data, report_path
):
    """Write a comprehensive text report."""
    lines = []
    def p(s=''):
        lines.append(s)

    p("=" * 72)
    p("NMF IDEAL-BASIS ANALYSIS REPORT (Downsampled)")
    p("=" * 72)
    p()

    # 1. Data summary
    p("1. DATA SUMMARY")
    p("-" * 40)
    p(f"  Image size:       {DS_SIZE[0]}x{DS_SIZE[1]} grayscale")
    p(f"  Feature vector:   {DS_SIZE[0] * DS_SIZE[1]} dimensions")
    p(f"  NMF components:   {nmf.n_components_}")
    p(f"  Total ideal imgs: {len(labels)}")
    for cls_name, cnt in counts.items():
        p(f"    {cls_name}: {cnt}")
    p()

    # 2. NMF training
    p("2. NMF TRAINING ON IDEAL IMAGES")
    p("-" * 40)
    p(f"  Reconstruction error: {nmf.reconstruction_err_:.4f}")
    p(f"  Iterations:           {nmf.n_iter_}")
    V_recon = W_ideal @ nmf.components_
    rel_error = np.linalg.norm(V_ideal - V_recon) / np.linalg.norm(V_ideal)
    p(f"  Relative Frobenius error: {rel_error:.4f} ({rel_error*100:.1f}%)")
    p()

    # 3. Component-class mapping
    p("3. COMPONENT-CLASS MAPPING")
    p("-" * 40)
    p()
    p("  (a) Each CLASS's dominant component:")
    for cls, (comp, score) in class_to_comp.items():
        p(f"      {cls:6s} -> C{comp+1}  (normalized weight = {score:.3f})")
    p()
    p("  (b) Each COMPONENT's dominant class:")
    for comp, (cls, score) in comp_to_class.items():
        p(f"      C{comp+1}     -> {cls:6s} (column fraction = {score:.3f})")
    p()
    p(f"  Unique component assignments: {n_unique} / {len(CLASS_LABELS)}")
    if n_unique == len(CLASS_LABELS):
        p("  GOOD: Each class maps to a DISTINCT component")
    else:
        p(f"  NOTE: Only {n_unique} distinct components for {len(CLASS_LABELS)} classes")
        comp_counts = Counter(class_comp_map.values())
        for comp_id, cnt in comp_counts.items():
            if cnt > 1:
                colliders = [CLASS_LABELS[c] for c, k in class_comp_map.items() if k == comp_id]
                p(f"    Component C{comp_id+1} shared by: {', '.join(colliders)}")
    p()

    # 4. Classification accuracy
    p("4. IDEAL IMAGE CLASSIFICATION ACCURACY")
    p("-" * 40)
    p(f"  Accuracy (dominant-component matching): {accuracy:.1%}")
    p()
    p("  Normalized weight matrix (rows=classes, cols=components):")
    header = "         " + "  ".join([f"  C{k+1}  " for k in range(nmf.n_components_)])
    p(header)
    for c in range(len(CLASS_LABELS)):
        row = f"  {CLASS_LABELS[c]:6s}  " + "  ".join([f"{normed[c,k]:.3f}" for k in range(nmf.n_components_)])
        p(row)
    p()

    # 5. Cosine similarity
    n_classes = len(CLASS_LABELS)
    centroids = np.zeros((n_classes, W_ideal.shape[1]))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = W_ideal[mask].mean(axis=0)
    sim = cosine_similarity(centroids)

    p("5. COSINE SIMILARITY BETWEEN CLASS CENTROIDS")
    p("-" * 40)
    header = "         " + "  ".join([f"{cl:>6s}" for cl in CLASS_LABELS])
    p(header)
    for i in range(n_classes):
        row = f"  {CLASS_LABELS[i]:6s}  " + "  ".join([f"{sim[i,j]:.3f}" for j in range(n_classes)])
        p(row)
    p()
    pairs = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            pairs.append((sim[i,j], CLASS_LABELS[i], CLASS_LABELS[j]))
    pairs.sort()
    p(f"  Most distinct pair:  {pairs[0][1]} vs {pairs[0][2]} (sim = {pairs[0][0]:.3f})")
    p(f"  Most similar pair:   {pairs[-1][1]} vs {pairs[-1][2]} (sim = {pairs[-1][0]:.3f})")
    p()

    # 6. Silhouette score
    if len(set(labels)) > 1:
        sil = silhouette_score(W_ideal, labels)
        p("6. CLUSTERING QUALITY")
        p("-" * 40)
        p(f"  Silhouette score (NMF space): {sil:.3f}")
        p(f"    (1.0 = perfect, 0 = overlapping, <0 = misassigned)")
        if sil > 0.5:
            p("    Good separation between reconstruction classes")
        elif sil > 0.25:
            p("    Moderate separation — some overlap between classes")
        else:
            p("    Poor separation — classes significantly overlap in NMF space")
        p()

    # 7. Trajectory analysis
    p("7. TRAJECTORY PROJECTION RESULTS")
    p("-" * 40)
    p()

    for traj_type, traj_list in TRAJECTORIES.items():
        p(f"  --- {traj_type} Trajectories ---")
        for traj_path, traj_name in traj_list:
            key = f"{traj_path.replace('/', '_')}_{traj_name}"
            if key not in traj_data:
                p(f"  {traj_name}: NO DATA")
                continue
            W = traj_data[key]['W']
            n_frames = traj_data[key]['n_frames']
            W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)
            avg = W_norm.mean(axis=0)

            p(f"  {traj_name} ({n_frames} frames):")
            for k in range(W.shape[1]):
                cls_label = comp_to_class.get(k, ('?', 0))[0]
                p(f"    C{k+1} ({cls_label:5s}): mean={avg[k]:.3f}, "
                  f"min={W_norm[:,k].min():.3f}, max={W_norm[:,k].max():.3f}")

            dominant = np.argmax(avg)
            p(f"    -> Dominant: C{dominant+1} ({comp_to_class.get(dominant, ('?',0))[0]}) "
              f"at {avg[dominant]:.1%}")
            p()

    # 8. HTR vs RT13 comparison
    p("8. HTR vs RT13 COMPARISON")
    p("-" * 40)
    p()

    avg_comp = {'HTR': [], 'RT13': []}
    for key, data in traj_data.items():
        W = data['W']
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)
        avg = W_norm.mean(axis=0)
        avg_comp[data['type']].append(avg)

    htr_mean = np.mean(avg_comp['HTR'], axis=0) if avg_comp['HTR'] else np.zeros(4)
    rt13_mean = np.mean(avg_comp['RT13'], axis=0) if avg_comp['RT13'] else np.zeros(4)

    n_comp = len(htr_mean)
    p("  Component    HTR_avg    RT13_avg   Diff(H-R)")
    for k in range(n_comp):
        cls_label = comp_to_class.get(k, ('?', 0))[0]
        diff = htr_mean[k] - rt13_mean[k]
        p(f"  C{k+1} ({cls_label:5s}) {htr_mean[k]:8.3f}   {rt13_mean[k]:8.3f}   {diff:+8.3f}")
    p()

    # 9. Consistency
    p("9. CONSISTENCY ASSESSMENT")
    p("-" * 40)
    p()

    for ttype in ['HTR', 'RT13']:
        if len(avg_comp[ttype]) >= 2:
            avgs_arr = np.array(avg_comp[ttype])
            within_sim = cosine_similarity(avgs_arr)
            n = len(avgs_arr)
            pair_sims = []
            for i in range(n):
                for j in range(i+1, n):
                    pair_sims.append(within_sim[i, j])
            mean_sim = np.mean(pair_sims) if pair_sims else 0
            p(f"  {ttype} within-group cosine similarity: {mean_sim:.3f}")
            if mean_sim > 0.95:
                p(f"    Highly consistent across {ttype} trajectories")
            elif mean_sim > 0.85:
                p(f"    Moderately consistent across {ttype} trajectories")
            else:
                p(f"    Low consistency across {ttype} trajectories")
        p()

    if avg_comp['HTR'] and avg_comp['RT13']:
        htr_centroid = np.mean(avg_comp['HTR'], axis=0).reshape(1, -1)
        rt13_centroid = np.mean(avg_comp['RT13'], axis=0).reshape(1, -1)
        between_sim = cosine_similarity(htr_centroid, rt13_centroid)[0, 0]
        p(f"  HTR vs RT13 between-group cosine similarity: {between_sim:.3f}")
        if between_sim < 0.85:
            p("    HTR and RT13 trajectories are DISTINGUISHABLE in NMF space")
        else:
            p("    HTR and RT13 trajectories are NOT well separated")
    p()

    p("=" * 72)
    p("END OF REPORT")
    p("=" * 72)

    report_text = '\n'.join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  Saved: {report_path.name}")

    return report_text


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Load ideal images
    print("=" * 60)
    print(f"STEP 1: Loading ideal images ({DS_SIZE[0]}x{DS_SIZE[1]} grayscale)")
    print("=" * 60)
    V_ideal, labels, counts = load_ideal_images()
    print(f"  Total: {len(labels)} images, {V_ideal.shape[1]} features each")

    # Step 2: Train NMF
    print("\n" + "=" * 60)
    print("STEP 2: Training NMF on ideal images")
    print("=" * 60)
    nmf, W_ideal, H = train_nmf(V_ideal, n_components=4, max_iter=1000)
    print(f"  Reconstruction error: {nmf.reconstruction_err_:.4f}")
    print(f"  Iterations: {nmf.n_iter_}")

    # Step 3: Analyze component-class mapping
    print("\n" + "=" * 60)
    print("STEP 3: Analyzing component-class mapping")
    print("=" * 60)
    means, stds = compute_class_means(W_ideal, labels, len(CLASSES))
    class_to_comp, comp_to_class, normed = component_class_mapping(means)
    accuracy, class_comp_map, n_unique = classification_accuracy(W_ideal, labels)

    print(f"  Classification accuracy: {accuracy:.1%}")
    print(f"  Unique assignments: {n_unique}/{len(CLASS_LABELS)}")
    for cls, (comp, score) in class_to_comp.items():
        print(f"    {cls} -> C{comp+1} ({score:.3f})")

    # Step 4: Ideal-image plots
    print("\n" + "=" * 60)
    print("STEP 4: Generating ideal image analysis plots")
    print("=" * 60)
    plot_basis_components(H, comp_to_class,
                          OUTPUT_DIR / "01_basis_components.png")
    plot_ideal_decomposition(means, stds, normed,
                              OUTPUT_DIR / "02_ideal_decomposition.png")
    plot_confusion_matrix(W_ideal, labels, class_comp_map,
                           OUTPUT_DIR / "03_confusion_matrix.png")
    plot_cosine_similarity(W_ideal, labels,
                            OUTPUT_DIR / "04_cosine_similarity.png")

    # Step 5: Project all trajectories
    print("\n" + "=" * 60)
    print("STEP 5: Projecting all 6 trajectories onto ideal basis")
    print("=" * 60)
    traj_data = {}
    for traj_type, traj_list in TRAJECTORIES.items():
        print(f"\n  --- {traj_type} ---")
        for traj_path, traj_name in traj_list:
            V_traj, files = load_trajectory(traj_path)
            if V_traj is not None and len(V_traj) > 0:
                W_traj = nmf.transform(V_traj)
                key = f"{traj_path.replace('/', '_')}_{traj_name}"
                traj_data[key] = {
                    'W': W_traj,
                    'n_frames': len(V_traj),
                    'type': traj_type,
                    'name': traj_name,
                }
                print(f"  {traj_name}: {len(V_traj)} frames projected")

    # Step 6: Trajectory comparison plots
    print("\n" + "=" * 60)
    print("STEP 6: Generating trajectory comparison plots")
    print("=" * 60)
    plot_trajectory_decomposition(traj_data, comp_to_class,
                                   OUTPUT_DIR / "05_trajectory_decomposition.png")
    plot_stacked_trajectories(traj_data, comp_to_class,
                               OUTPUT_DIR / "06_stacked_composition.png")
    plot_htr_vs_rt13_summary(traj_data, comp_to_class,
                              OUTPUT_DIR / "07_htr_vs_rt13_summary.png")

    # Step 7: Generate report
    print("\n" + "=" * 60)
    print("STEP 7: Generating analysis report")
    print("=" * 60)
    report = generate_report(
        counts, nmf, V_ideal, W_ideal, labels, means, stds, normed,
        class_to_comp, comp_to_class, accuracy, class_comp_map, n_unique,
        traj_data, OUTPUT_DIR / "REPORT.txt"
    )

    print("\n" + report)
    print(f"\nAll results saved to: {OUTPUT_DIR}")

    return nmf, traj_data, comp_to_class


if __name__ == "__main__":
    nmf, traj_data, comp_to_class = main()
