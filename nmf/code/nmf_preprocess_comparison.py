"""
NMF Group Analysis — Preprocessing Comparison

Compare different per-frame normalization strategies to align
2022 and 2025 RHEED images before running separate-group NMF.

Methods tested:
  1. Raw (baseline) — just grayscale + downsample
  2. Histogram equalization — flatten each frame's histogram
  3. CLAHE — adaptive local contrast normalization
  4. Percentile normalization — clip [p2, p98], rescale to [0,1]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"
OUTPUT_DIR = Path(__file__).parent / "nmf_preprocess_comparison"

IMG_SIZE = (128, 96)
MASK_RECT = [56, 40, 62, 50]
N_COMPONENTS = 3
MAX_ITER = 1000

TRAJECTORIES = OrderedDict({
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
})

COLORS_3 = ['#e41a1c', '#377eb8', '#4daf4a']


# ─── Preprocessing methods ────────────────────────────────────────────────────

def _base_load(img_path):
    """Load image as grayscale PIL, resize, mask specular."""
    img = Image.open(img_path).convert('L')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    draw.rectangle(MASK_RECT, fill=0)
    return img


def preprocess_raw(img_path):
    """Method 1: Raw grayscale, just normalize to [0,1]."""
    img = _base_load(img_path)
    return np.array(img, dtype=np.float64) / 255.0


def preprocess_histeq(img_path):
    """Method 2: Histogram equalization per frame."""
    img = _base_load(img_path)
    img = ImageOps.equalize(img)
    return np.array(img, dtype=np.float64) / 255.0


def preprocess_clahe(img_path):
    """Method 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    try:
        import cv2
        img = _base_load(img_path)
        arr = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        return arr.astype(np.float64) / 255.0
    except ImportError:
        # Fallback: use PIL autocontrast as approximation
        img = _base_load(img_path)
        img = ImageOps.autocontrast(img, cutoff=2)
        return np.array(img, dtype=np.float64) / 255.0


def preprocess_percentile(img_path):
    """Method 4: Percentile clipping [p2, p98] + rescale to [0,1]."""
    img = _base_load(img_path)
    arr = np.array(img, dtype=np.float64)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    arr = np.clip(arr, p2, p98)
    rng = p98 - p2
    if rng > 0:
        arr = (arr - p2) / rng
    else:
        arr = arr / 255.0
    return arr


METHODS = OrderedDict({
    'Raw':        preprocess_raw,
    'HistEq':     preprocess_histeq,
    'CLAHE':      preprocess_clahe,
    'Percentile': preprocess_percentile,
})


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_trajectory(traj_path, preprocess_fn):
    """Load trajectory with given preprocessing."""
    traj_dir = TRAJECTORY_DIR / traj_path
    if not traj_dir.exists():
        return None, 0

    files = sorted(list(traj_dir.glob('*.bmp')) + list(traj_dir.glob('*.png')))
    arrays = []
    for f in files:
        try:
            arr = preprocess_fn(f)
            arrays.append(arr.flatten())
        except:
            pass

    if not arrays:
        return None, 0
    return np.array(arrays), len(arrays)


# ─── Analysis ─────────────────────────────────────────────────────────────────

def run_group_nmf_for_method(method_name, preprocess_fn):
    """Run separate-group NMF with a given preprocessing method."""
    results = {}

    for group_name, traj_list in TRAJECTORIES.items():
        traj_data = []
        all_rows = []

        for traj_path, traj_name in traj_list:
            V, n = load_trajectory(traj_path, preprocess_fn)
            if V is not None:
                traj_data.append((traj_name, V, n))
                all_rows.append(V)

        if not all_rows:
            continue

        V_pooled = np.vstack(all_rows)

        model = NMF(n_components=N_COMPONENTS, init='nndsvda',
                     max_iter=MAX_ITER, random_state=42)
        W_pooled = model.fit_transform(V_pooled)
        H = model.components_

        V_recon = W_pooled @ H
        rel_err = np.linalg.norm(V_pooled - V_recon) / (np.linalg.norm(V_pooled) + 1e-10)

        # Split W per trajectory
        offset = 0
        per_traj = []
        for traj_name, V_traj, n in traj_data:
            W_traj = W_pooled[offset:offset + n]
            offset += n
            W_norm = W_traj / (W_traj.sum(axis=1, keepdims=True) + 1e-10)
            per_traj.append({
                'name': traj_name,
                'W': W_traj,
                'W_norm': W_norm,
                'n_frames': n,
                'avg': W_norm.mean(axis=0),
            })

        results[group_name] = {
            'H': H,
            'rel_err': rel_err,
            'n_iter': model.n_iter_,
            'trajectories': per_traj,
        }

    return results


def compute_consistency(results):
    """Compute within-group consistency metrics."""
    metrics = {}
    for group_name, data in results.items():
        avgs = [t['avg'] for t in data['trajectories']]
        names = [t['name'] for t in data['trajectories']]

        if len(avgs) < 2:
            continue

        avgs_arr = np.array(avgs)
        sim = cosine_similarity(avgs_arr)

        pair_info = []
        for i in range(len(avgs)):
            for j in range(i + 1, len(avgs)):
                pair_info.append({
                    'a': names[i], 'b': names[j],
                    'sim': sim[i, j],
                })

        mean_sim = np.mean([p['sim'] for p in pair_info])

        # Also check 2022 vs 2025 split specifically
        cross_year = []
        same_year = []
        for p in pair_info:
            a_is_2022 = p['a'].startswith('RR')
            b_is_2022 = p['b'].startswith('RR')
            if a_is_2022 == b_is_2022:
                same_year.append(p['sim'])
            else:
                cross_year.append(p['sim'])

        metrics[group_name] = {
            'mean_sim': mean_sim,
            'pairs': pair_info,
            'rel_err': data['rel_err'],
            'same_year_sim': np.mean(same_year) if same_year else None,
            'cross_year_sim': np.mean(cross_year) if cross_year else None,
        }

    return metrics


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_method_comparison_summary(all_metrics, output_path):
    """Bar chart comparing all methods on key metrics."""
    methods = list(all_metrics.keys())
    groups = ['HTR', 'RT13']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Overall within-group similarity
    ax = axes[0, 0]
    x = np.arange(len(methods))
    width = 0.35
    for i, g in enumerate(groups):
        vals = [all_metrics[m].get(g, {}).get('mean_sim', 0) for m in methods]
        ax.bar(x + i * width, vals, width, label=g, alpha=0.85)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Within-Group Consistency (higher = better)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 2: Cross-year similarity
    ax = axes[0, 1]
    for i, g in enumerate(groups):
        vals = [all_metrics[m].get(g, {}).get('cross_year_sim', 0) or 0
                for m in methods]
        ax.bar(x + i * width, vals, width, label=g, alpha=0.85)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Cross-Year Consistency (2022 vs 2025, higher = better)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 3: Same-year similarity
    ax = axes[1, 0]
    for i, g in enumerate(groups):
        vals = [all_metrics[m].get(g, {}).get('same_year_sim', 0) or 0
                for m in methods]
        ax.bar(x + i * width, vals, width, label=g, alpha=0.85)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Same-Year Consistency (higher = better)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 4: Reconstruction error
    ax = axes[1, 1]
    for i, g in enumerate(groups):
        vals = [all_metrics[m].get(g, {}).get('rel_err', 0) for m in methods]
        ax.bar(x + i * width, vals, width, label=g, alpha=0.85)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Relative Error')
    ax.set_title('Reconstruction Error (lower = better fit)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Preprocessing Method Comparison — Group NMF', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_per_method_basis(all_results, output_path):
    """Basis components for each method, HTR and RT13 side by side."""
    n_methods = len(all_results)

    fig, axes = plt.subplots(n_methods * 2, N_COMPONENTS,
                              figsize=(5 * N_COMPONENTS, 4 * n_methods * 2))

    row = 0
    for method_name, results in all_results.items():
        for group_name in ['HTR', 'RT13']:
            if group_name not in results:
                row += 1
                continue
            H = results[group_name]['H']
            rel_err = results[group_name]['rel_err']

            for k in range(N_COMPONENTS):
                ax = axes[row, k]
                comp = H[k].reshape(IMG_SIZE[1], IMG_SIZE[0])
                comp = comp / (comp.max() + 1e-10)
                ax.imshow(comp, cmap='hot', interpolation='nearest')
                if k == 0:
                    ax.set_ylabel(f'{method_name}\n{group_name}\nerr={rel_err:.1%}',
                                  fontsize=10, fontweight='bold')
                if row == 0:
                    ax.set_title(f'C{k+1}', fontsize=13)
                ax.axis('off')
            row += 1

    plt.suptitle('Basis Components by Preprocessing Method\n'
                 f'({N_COMPONENTS} components, 128x96 grayscale)', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_per_method_evolution(all_results, output_path):
    """Evolution curves for each method — one row per method, 6 columns."""
    n_methods = len(all_results)

    fig, axes = plt.subplots(n_methods, 6, figsize=(30, 5 * n_methods))

    for m_idx, (method_name, results) in enumerate(all_results.items()):
        col = 0
        for group_name, traj_list in TRAJECTORIES.items():
            if group_name not in results:
                col += len(traj_list)
                continue

            data = results[group_name]
            for t_idx, traj in enumerate(data['trajectories']):
                ax = axes[m_idx, col]
                W_norm = traj['W_norm']

                for k in range(N_COMPONENTS):
                    ax.plot(W_norm[:, k], color=COLORS_3[k],
                            label=f'C{k+1}', linewidth=1.0, alpha=0.85)

                title = f"{traj['name']} ({group_name})"
                if col == 0:
                    title = f"[{method_name}] {title}"
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, alpha=0.3)
                if t_idx == 0:
                    ax.legend(fontsize=7, loc='best')
                col += 1

    plt.suptitle('Component Evolution by Preprocessing Method\n'
                 'Cols 1-3: HTR | Cols 4-6: RT13', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_consistency_bars(all_metrics, output_path):
    """Per-pair consistency across methods."""
    methods = list(all_metrics.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for g_idx, group_name in enumerate(['HTR', 'RT13']):
        ax = axes[g_idx]

        # Collect all pairs for this group
        pair_labels = None
        all_sims = {}

        for method_name in methods:
            m = all_metrics[method_name].get(group_name)
            if m is None:
                continue
            pairs = m['pairs']
            if pair_labels is None:
                pair_labels = [f"{p['a']}\nvs\n{p['b']}" for p in pairs]
            all_sims[method_name] = [p['sim'] for p in pairs]

        if pair_labels is None:
            continue

        x = np.arange(len(pair_labels))
        n_methods = len(all_sims)
        width = 0.8 / n_methods

        for i, (method_name, sims) in enumerate(all_sims.items()):
            ax.bar(x + i * width, sims, width, label=method_name, alpha=0.85)

        ax.set_xticks(x + width * (n_methods - 1) / 2)
        ax.set_xticklabels(pair_labels, fontsize=9)
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'{group_name} — Pairwise Consistency', fontsize=13,
                     fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Pairwise Trajectory Consistency by Preprocessing Method', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_sample_frames(output_path):
    """Show same frame preprocessed with each method for visual comparison."""
    # Pick one frame from 2022 and one from 2025
    sample_frames = [
        (TRAJECTORY_DIR / '2025-10-04/A', 'HL251004A (2025)'),
        (TRAJECTORY_DIR / '2022-02-06', 'RR220206A (2022)'),
    ]

    fig, axes = plt.subplots(len(sample_frames), len(METHODS) + 1,
                              figsize=(4 * (len(METHODS) + 1), 4 * len(sample_frames)))

    for row, (traj_dir, label) in enumerate(sample_frames):
        files = sorted(list(traj_dir.glob('*.bmp')) + list(traj_dir.glob('*.png')))
        mid = files[len(files) // 2]

        # Original
        ax = axes[row, 0]
        img = Image.open(mid).convert('L')
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        ax.imshow(np.array(img), cmap='gray', vmin=0, vmax=255)
        ax.set_title('Original' if row == 0 else '', fontsize=11)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.axis('off')
        # Show stats
        arr = np.array(img, dtype=np.float64)
        ax.text(0.02, 0.02, f'μ={arr.mean():.0f} σ={arr.std():.0f}',
                transform=ax.transAxes, fontsize=9, color='yellow',
                backgroundcolor='black')

        # Each method
        for col, (method_name, fn) in enumerate(METHODS.items(), 1):
            ax = axes[row, col]
            arr = fn(mid)
            ax.imshow(arr.reshape(IMG_SIZE[1], IMG_SIZE[0]),
                      cmap='gray', vmin=0, vmax=1)
            ax.set_title(method_name if row == 0 else '', fontsize=11)
            ax.axis('off')
            ax.text(0.02, 0.02, f'μ={arr.mean():.2f} σ={arr.std():.2f}',
                    transform=ax.transAxes, fontsize=9, color='yellow',
                    backgroundcolor='black')

    plt.suptitle('Sample Frame Comparison: 2025 vs 2022 under each preprocessing', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# ─── Report ───────────────────────────────────────────────────────────────────

def generate_report(all_metrics, report_path):
    """Text report comparing all methods."""
    lines = []
    def p(s=''):
        lines.append(s)

    p("=" * 72)
    p("PREPROCESSING COMPARISON REPORT — GROUP NMF")
    p("=" * 72)
    p()
    p(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]} grayscale")
    p(f"Components: {N_COMPONENTS}")
    p(f"Methods tested: {', '.join(all_metrics.keys())}")
    p()

    for group_name in ['HTR', 'RT13']:
        p(f"{'='*72}")
        p(f"  {group_name} GROUP")
        p(f"{'='*72}")
        p()

        # Summary table
        p(f"  {'Method':<14} {'Rel.Err':>8} {'Overall':>8} {'SameYr':>8} {'CrossYr':>8}")
        p(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for method_name, metrics in all_metrics.items():
            m = metrics.get(group_name)
            if m is None:
                continue
            re = m['rel_err']
            ov = m['mean_sim']
            sy = m['same_year_sim']
            cy = m['cross_year_sim']
            sy_str = f"{sy:.3f}" if sy is not None else "  N/A "
            cy_str = f"{cy:.3f}" if cy is not None else "  N/A "
            p(f"  {method_name:<14} {re:8.4f} {ov:8.3f} {sy_str:>8} {cy_str:>8}")
        p()

        # Per-pair details
        p(f"  Pairwise details:")
        for method_name, metrics in all_metrics.items():
            m = metrics.get(group_name)
            if m is None:
                continue
            p(f"    {method_name}:")
            for pair in m['pairs']:
                p(f"      {pair['a']} vs {pair['b']}: {pair['sim']:.3f}")
        p()

    # Best method determination
    p("=" * 72)
    p("SUMMARY")
    p("=" * 72)
    p()

    # Rank by cross-year consistency
    p("  Ranking by cross-year consistency (average of HTR + RT13):")
    scores = []
    for method_name, metrics in all_metrics.items():
        cy_vals = []
        for g in ['HTR', 'RT13']:
            m = metrics.get(g)
            if m and m['cross_year_sim'] is not None:
                cy_vals.append(m['cross_year_sim'])
        avg_cy = np.mean(cy_vals) if cy_vals else 0
        scores.append((method_name, avg_cy))
    scores.sort(key=lambda x: -x[1])

    for rank, (method, score) in enumerate(scores, 1):
        marker = " <-- BEST" if rank == 1 else ""
        p(f"    {rank}. {method}: {score:.3f}{marker}")
    p()

    # Also rank by overall consistency
    p("  Ranking by overall within-group consistency:")
    scores2 = []
    for method_name, metrics in all_metrics.items():
        ov_vals = []
        for g in ['HTR', 'RT13']:
            m = metrics.get(g)
            if m:
                ov_vals.append(m['mean_sim'])
        avg_ov = np.mean(ov_vals) if ov_vals else 0
        scores2.append((method_name, avg_ov))
    scores2.sort(key=lambda x: -x[1])

    for rank, (method, score) in enumerate(scores2, 1):
        marker = " <-- BEST" if rank == 1 else ""
        p(f"    {rank}. {method}: {score:.3f}{marker}")
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

    # Show sample frames first
    print("Generating sample frame comparison...")
    plot_sample_frames(OUTPUT_DIR / "00_sample_frames.png")

    # Run NMF for each method
    all_results = OrderedDict()
    all_metrics = OrderedDict()

    for method_name, preprocess_fn in METHODS.items():
        print(f"\n{'='*65}")
        print(f"Running group NMF with preprocessing: {method_name}")
        print(f"{'='*65}")

        results = run_group_nmf_for_method(method_name, preprocess_fn)
        metrics = compute_consistency(results)

        all_results[method_name] = results
        all_metrics[method_name] = metrics

        # Print summary
        for group_name, m in metrics.items():
            print(f"  {group_name}: err={m['rel_err']:.4f}, "
                  f"consistency={m['mean_sim']:.3f}, "
                  f"cross-year={m['cross_year_sim']:.3f}" if m['cross_year_sim'] else
                  f"  {group_name}: err={m['rel_err']:.4f}, "
                  f"consistency={m['mean_sim']:.3f}")

    # Generate all plots
    print(f"\n{'='*65}")
    print("Generating comparison plots...")
    print(f"{'='*65}")

    plot_method_comparison_summary(all_metrics,
                                   OUTPUT_DIR / "01_method_comparison.png")
    plot_per_method_basis(all_results,
                           OUTPUT_DIR / "02_basis_by_method.png")
    plot_per_method_evolution(all_results,
                               OUTPUT_DIR / "03_evolution_by_method.png")
    plot_consistency_bars(all_metrics,
                           OUTPUT_DIR / "04_pairwise_consistency.png")

    # Generate report
    print(f"\n{'='*65}")
    print("Generating report...")
    print(f"{'='*65}")
    report = generate_report(all_metrics, OUTPUT_DIR / "REPORT.txt")
    print("\n" + report)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    return all_results, all_metrics


if __name__ == "__main__":
    all_results, all_metrics = main()
