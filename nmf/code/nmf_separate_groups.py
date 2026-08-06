"""
NMF Separate Groups Analysis

Two approaches compared:
  A) Separate-group NMF: pool all HTR trajectories → shared HTR basis,
     pool all RT13 trajectories → shared RT13 basis.
     Then project each trajectory onto its group's basis.
  B) Per-trajectory NMF: run NMF independently on each of the 6 trajectories.

For each approach, we analyze:
  - Basis components (what do the learned patterns look like?)
  - Component evolution per trajectory (W over time)
  - Within-group consistency (do same-type trajectories evolve similarly?)
  - Reconstruction quality
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"
OUTPUT_DIR = Path(__file__).parent / "nmf_separate_results"

# Image preprocessing
IMG_SIZE = (128, 96)   # width, height — moderate downsample
MASK_RECT = [56, 40, 62, 50]  # specular mask scaled to 128x96

N_COMPONENTS = 3  # components per NMF
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
COLORS_6 = ['#e41a1c', '#ff7f00', '#984ea3', '#377eb8', '#4daf4a', '#a65628']


# ─── Preprocessing ────────────────────────────────────────────────────────────

def load_and_preprocess(img_path):
    """Load → grayscale → resize → mask specular → normalize to [0,1]."""
    img = Image.open(img_path).convert('L')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    draw.rectangle(MASK_RECT, fill=0)
    return np.array(img, dtype=np.float64) / 255.0


def load_trajectory(traj_path):
    """Load all frames from a trajectory, return (matrix, n_frames)."""
    traj_dir = TRAJECTORY_DIR / traj_path
    if not traj_dir.exists():
        print(f"  WARNING: {traj_dir} not found")
        return None, 0

    files = sorted(list(traj_dir.glob('*.bmp')) + list(traj_dir.glob('*.png')))
    arrays = []
    for f in files:
        try:
            arr = load_and_preprocess(f)
            arrays.append(arr.flatten())
        except:
            pass

    if not arrays:
        return None, 0
    return np.array(arrays), len(arrays)


# ─── NMF helpers ──────────────────────────────────────────────────────────────

def run_nmf(V, n_components=N_COMPONENTS, max_iter=MAX_ITER):
    """Fit NMF and return (model, W, H, recon_error)."""
    model = NMF(n_components=n_components, init='nndsvda',
                max_iter=max_iter, random_state=42)
    W = model.fit_transform(V)
    H = model.components_
    V_recon = W @ H
    rel_err = np.linalg.norm(V - V_recon) / (np.linalg.norm(V) + 1e-10)
    return model, W, H, rel_err


def normalize_weights(W):
    """Row-normalize W so each frame sums to 1."""
    return W / (W.sum(axis=1, keepdims=True) + 1e-10)


# ─── APPROACH A: Separate-Group NMF ──────────────────────────────────────────

def run_group_nmf():
    """Pool trajectories by type, train one NMF per group, project each."""
    print("=" * 65)
    print("APPROACH A: Separate-Group NMF")
    print("  (pool HTR together, pool RT13 together, shared basis per group)")
    print("=" * 65)

    group_results = {}

    for group_name, traj_list in TRAJECTORIES.items():
        print(f"\n--- {group_name} group ({len(traj_list)} trajectories) ---")

        # Load all trajectories in this group
        traj_matrices = []  # (name, V, n_frames)
        all_rows = []
        for traj_path, traj_name in traj_list:
            V, n = load_trajectory(traj_path)
            if V is not None:
                traj_matrices.append((traj_name, V, n))
                all_rows.append(V)
                print(f"  {traj_name}: {n} frames")

        if not all_rows:
            continue

        # Pool all frames into one matrix
        V_pooled = np.vstack(all_rows)
        print(f"  Pooled: {V_pooled.shape[0]} total frames, {V_pooled.shape[1]} features")

        # Train NMF on pooled data
        model, W_pooled, H, rel_err = run_nmf(V_pooled)
        print(f"  NMF: {N_COMPONENTS} components, rel. error = {rel_err:.4f} ({rel_err*100:.1f}%)")
        print(f"  Iterations: {model.n_iter_}")

        # Split W back per trajectory and also project (should be same, but let's be explicit)
        offset = 0
        per_traj = []
        for traj_name, V_traj, n in traj_matrices:
            W_traj = W_pooled[offset:offset + n]
            offset += n
            per_traj.append({
                'name': traj_name,
                'W': W_traj,
                'n_frames': n,
            })
            W_norm = normalize_weights(W_traj)
            avg = W_norm.mean(axis=0)
            print(f"  {traj_name} avg composition: " +
                  ", ".join([f"C{k+1}={avg[k]:.3f}" for k in range(N_COMPONENTS)]))

        group_results[group_name] = {
            'model': model,
            'H': H,
            'rel_err': rel_err,
            'trajectories': per_traj,
        }

    return group_results


# ─── APPROACH B: Per-Trajectory NMF ──────────────────────────────────────────

def run_per_trajectory_nmf():
    """Run independent NMF on each trajectory."""
    print("\n" + "=" * 65)
    print("APPROACH B: Per-Trajectory NMF")
    print("  (independent NMF on each of 6 trajectories)")
    print("=" * 65)

    per_traj_results = {}

    for group_name, traj_list in TRAJECTORIES.items():
        print(f"\n--- {group_name} trajectories ---")
        for traj_path, traj_name in traj_list:
            V, n = load_trajectory(traj_path)
            if V is None:
                continue

            model, W, H, rel_err = run_nmf(V)
            print(f"  {traj_name}: {n} frames, rel. error = {rel_err:.4f} ({rel_err*100:.1f}%), "
                  f"iter = {model.n_iter_}")

            W_norm = normalize_weights(W)
            avg = W_norm.mean(axis=0)
            print(f"    avg composition: " +
                  ", ".join([f"C{k+1}={avg[k]:.3f}" for k in range(N_COMPONENTS)]))

            per_traj_results[traj_name] = {
                'group': group_name,
                'model': model,
                'W': W,
                'H': H,
                'n_frames': n,
                'rel_err': rel_err,
            }

    return per_traj_results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_group_basis(group_results, output_path):
    """Side-by-side basis components for HTR vs RT13 group NMF."""
    fig, axes = plt.subplots(2, N_COMPONENTS, figsize=(5 * N_COMPONENTS, 9))

    for row, (group_name, data) in enumerate(group_results.items()):
        H = data['H']
        rel_err = data['rel_err']
        for k in range(N_COMPONENTS):
            ax = axes[row, k]
            comp = H[k].reshape(IMG_SIZE[1], IMG_SIZE[0])
            comp = comp / (comp.max() + 1e-10)
            ax.imshow(comp, cmap='hot', interpolation='nearest')
            ax.set_title(f'{group_name} C{k+1}', fontsize=13)
            ax.axis('off')
        # Add error annotation on first panel
        axes[row, 0].text(0.02, 0.02, f'err={rel_err:.1%}',
                          transform=axes[row, 0].transAxes, fontsize=10,
                          color='white', backgroundcolor='black')

    plt.suptitle('Group NMF Basis Components\n'
                 f'Top: HTR basis | Bottom: RT13 basis ({N_COMPONENTS} components each)',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_group_evolution(group_results, output_path):
    """Component evolution for each trajectory under its group basis."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for row, (group_name, data) in enumerate(group_results.items()):
        for col, traj in enumerate(data['trajectories']):
            ax = axes[row, col]
            W_norm = normalize_weights(traj['W'])

            for k in range(N_COMPONENTS):
                ax.plot(W_norm[:, k], color=COLORS_3[k],
                        label=f'C{k+1}', linewidth=1.2, alpha=0.85)

            ax.set_title(f"{traj['name']} ({group_name})",
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Normalized Weight')
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Group NMF: Component Evolution per Trajectory\n'
                 'Top: HTR trajectories (shared HTR basis) | '
                 'Bottom: RT13 trajectories (shared RT13 basis)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_group_stacked(group_results, output_path):
    """Stacked area plots under group basis."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for row, (group_name, data) in enumerate(group_results.items()):
        for col, traj in enumerate(data['trajectories']):
            ax = axes[row, col]
            W_norm = normalize_weights(traj['W'])

            ax.stackplot(range(len(W_norm)),
                         [W_norm[:, k] for k in range(N_COMPONENTS)],
                         labels=[f'C{k+1}' for k in range(N_COMPONENTS)],
                         colors=COLORS_3[:N_COMPONENTS], alpha=0.75)

            ax.set_title(f"{traj['name']} ({group_name})",
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Composition')
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('Group NMF: Composition (Stacked)\n'
                 'Top: HTR (shared basis) | Bottom: RT13 (shared basis)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_per_traj_basis(per_traj_results, output_path):
    """Basis components for each of the 6 independent NMFs."""
    fig, axes = plt.subplots(6, N_COMPONENTS, figsize=(5 * N_COMPONENTS, 26))

    traj_order = []
    for group_name, traj_list in TRAJECTORIES.items():
        for _, traj_name in traj_list:
            traj_order.append((traj_name, group_name))

    for row, (traj_name, group_name) in enumerate(traj_order):
        if traj_name not in per_traj_results:
            continue
        data = per_traj_results[traj_name]
        H = data['H']
        rel_err = data['rel_err']

        for k in range(N_COMPONENTS):
            ax = axes[row, k]
            comp = H[k].reshape(IMG_SIZE[1], IMG_SIZE[0])
            comp = comp / (comp.max() + 1e-10)
            ax.imshow(comp, cmap='hot', interpolation='nearest')
            if k == 0:
                ax.set_ylabel(f'{traj_name}\n({group_name})', fontsize=11,
                              fontweight='bold')
            ax.set_title(f'C{k+1}' if row == 0 else '', fontsize=12)
            ax.axis('off')

        axes[row, 0].text(0.02, 0.02, f'err={rel_err:.1%}',
                          transform=axes[row, 0].transAxes, fontsize=9,
                          color='white', backgroundcolor='black')

    plt.suptitle(f'Per-Trajectory NMF: Basis Components ({N_COMPONENTS} each)\n'
                 'Each row is an independent NMF on one trajectory', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_per_traj_evolution(per_traj_results, output_path):
    """Component evolution for each trajectory's own NMF."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for row, (group_name, traj_list) in enumerate(TRAJECTORIES.items()):
        for col, (_, traj_name) in enumerate(traj_list):
            ax = axes[row, col]
            if traj_name not in per_traj_results:
                continue

            data = per_traj_results[traj_name]
            W_norm = normalize_weights(data['W'])

            for k in range(N_COMPONENTS):
                ax.plot(W_norm[:, k], color=COLORS_3[k],
                        label=f'C{k+1}', linewidth=1.2, alpha=0.85)

            ax.set_title(f"{traj_name} ({group_name})",
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Normalized Weight')
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Trajectory NMF: Component Evolution\n'
                 'Top: HTR | Bottom: RT13 (each has its OWN basis)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_consistency_comparison(group_results, per_traj_results, output_path):
    """Compare within-group consistency: group NMF vs per-trajectory NMF."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (group_name, data) in enumerate(group_results.items()):
        # --- Group NMF consistency (left column) ---
        ax = axes[row, 0]
        trajs = data['trajectories']
        avgs = []
        names = []
        for t in trajs:
            W_norm = normalize_weights(t['W'])
            avg = W_norm.mean(axis=0)
            avgs.append(avg)
            names.append(t['name'])

        x = np.arange(N_COMPONENTS)
        width = 0.8 / len(trajs)
        for i, (name, avg) in enumerate(zip(names, avgs)):
            ax.bar(x + i * width, avg, width, label=name, alpha=0.85)

        ax.set_xticks(x + width * (len(trajs) - 1) / 2)
        ax.set_xticklabels([f'C{k+1}' for k in range(N_COMPONENTS)])
        ax.set_ylabel('Mean Normalized Weight')
        ax.set_title(f'{group_name} — Group NMF\n(shared basis, comparable)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Compute pairwise cosine similarity
        avgs_arr = np.array(avgs)
        sim = cosine_similarity(avgs_arr)
        pair_sims = []
        for i in range(len(avgs)):
            for j in range(i+1, len(avgs)):
                pair_sims.append(sim[i, j])
        mean_sim = np.mean(pair_sims) if pair_sims else 0
        ax.text(0.02, 0.98, f'Within-group sim: {mean_sim:.3f}',
                transform=ax.transAxes, fontsize=10, va='top',
                backgroundcolor='lightyellow')

        # --- Per-trajectory NMF (right column) ---
        ax = axes[row, 1]
        traj_list = TRAJECTORIES[group_name]
        avgs2 = []
        names2 = []
        for _, traj_name in traj_list:
            if traj_name in per_traj_results:
                W_norm = normalize_weights(per_traj_results[traj_name]['W'])
                avg = W_norm.mean(axis=0)
                avgs2.append(avg)
                names2.append(traj_name)

        for i, (name, avg) in enumerate(zip(names2, avgs2)):
            ax.bar(x + i * width, avg, width, label=name, alpha=0.85)

        ax.set_xticks(x + width * (len(names2) - 1) / 2)
        ax.set_xticklabels([f'C{k+1}' for k in range(N_COMPONENTS)])
        ax.set_ylabel('Mean Normalized Weight')
        ax.set_title(f'{group_name} — Per-Trajectory NMF\n(independent bases, NOT directly comparable)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        if len(avgs2) >= 2:
            avgs2_arr = np.array(avgs2)
            sim2 = cosine_similarity(avgs2_arr)
            pair_sims2 = []
            for i in range(len(avgs2)):
                for j in range(i+1, len(avgs2)):
                    pair_sims2.append(sim2[i, j])
            mean_sim2 = np.mean(pair_sims2) if pair_sims2 else 0
            ax.text(0.02, 0.98, f'Avg composition sim: {mean_sim2:.3f}\n(components not aligned!)',
                    transform=ax.transAxes, fontsize=10, va='top',
                    backgroundcolor='lightyellow')

    plt.suptitle('Within-Group Consistency: Group NMF vs Per-Trajectory NMF', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_evolution_overlay(group_results, output_path):
    """Overlay normalized evolution curves from same-group trajectories.
    Resample to common x-axis [0, 1] for fair comparison."""
    fig, axes = plt.subplots(2, N_COMPONENTS, figsize=(6 * N_COMPONENTS, 9))

    line_styles = ['-', '--', ':']

    for row, (group_name, data) in enumerate(group_results.items()):
        trajs = data['trajectories']
        for k in range(N_COMPONENTS):
            ax = axes[row, k]
            for t_idx, traj in enumerate(trajs):
                W_norm = normalize_weights(traj['W'])
                # Resample to [0, 1] normalized timeline
                n = len(W_norm)
                x_norm = np.linspace(0, 1, n)
                ax.plot(x_norm, W_norm[:, k],
                        color=COLORS_3[t_idx % len(COLORS_3)],
                        linestyle=line_styles[t_idx % len(line_styles)],
                        label=traj['name'], linewidth=1.5, alpha=0.8)

            ax.set_title(f'{group_name} — C{k+1}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Normalized Time (0→1)')
            ax.set_ylabel('Weight')
            ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Group NMF: Evolution Overlay (normalized time axis)\n'
                 'Top: HTR trajectories | Bottom: RT13 trajectories\n'
                 'Similar curves = consistent evolution pattern', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_basis_similarity_heatmap(group_results, output_path):
    """Cross-similarity between HTR and RT13 basis components."""
    H_htr = group_results['HTR']['H']
    H_rt13 = group_results['RT13']['H']

    sim = cosine_similarity(H_htr, H_rt13)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap='RdYlGn', vmin=-0.1, vmax=1)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    for i in range(N_COMPONENTS):
        for j in range(N_COMPONENTS):
            color = 'white' if sim[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{sim[i,j]:.3f}', ha='center', va='center',
                    fontsize=13, color=color, fontweight='bold')

    ax.set_xticks(range(N_COMPONENTS))
    ax.set_xticklabels([f'RT13 C{k+1}' for k in range(N_COMPONENTS)])
    ax.set_yticks(range(N_COMPONENTS))
    ax.set_yticklabels([f'HTR C{k+1}' for k in range(N_COMPONENTS)])
    ax.set_title('Cross-Similarity: HTR vs RT13 Basis Components\n'
                 '(high = shared pattern, low = unique to one group)', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# ─── Report ───────────────────────────────────────────────────────────────────

def generate_report(group_results, per_traj_results, report_path):
    """Comprehensive text report."""
    lines = []
    def p(s=''):
        lines.append(s)

    p("=" * 72)
    p("NMF SEPARATE GROUPS vs PER-TRAJECTORY COMPARISON REPORT")
    p("=" * 72)
    p()
    p(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]} grayscale ({IMG_SIZE[0]*IMG_SIZE[1]} features)")
    p(f"Components: {N_COMPONENTS}")
    p()

    # ── Group NMF ──
    p("=" * 72)
    p("APPROACH A: SEPARATE-GROUP NMF")
    p("=" * 72)
    p()

    for group_name, data in group_results.items():
        p(f"--- {group_name} Group ---")
        p(f"  Relative reconstruction error: {data['rel_err']:.4f} ({data['rel_err']*100:.1f}%)")
        p()

        avgs = []
        for traj in data['trajectories']:
            W_norm = normalize_weights(traj['W'])
            avg = W_norm.mean(axis=0)
            avgs.append(avg)

            p(f"  {traj['name']} ({traj['n_frames']} frames):")
            for k in range(N_COMPONENTS):
                p(f"    C{k+1}: mean={avg[k]:.3f}, "
                  f"std={W_norm[:,k].std():.3f}, "
                  f"range=[{W_norm[:,k].min():.3f}, {W_norm[:,k].max():.3f}]")
            p()

        # Within-group consistency
        if len(avgs) >= 2:
            avgs_arr = np.array(avgs)
            sim = cosine_similarity(avgs_arr)
            pair_sims = []
            for i in range(len(avgs)):
                for j in range(i+1, len(avgs)):
                    pair_sims.append(sim[i, j])
                    p(f"  Similarity {data['trajectories'][i]['name']} vs "
                      f"{data['trajectories'][j]['name']}: {sim[i,j]:.3f}")
            mean_sim = np.mean(pair_sims)
            p(f"  Mean within-group similarity: {mean_sim:.3f}")
            if mean_sim > 0.95:
                p(f"  -> Highly consistent")
            elif mean_sim > 0.85:
                p(f"  -> Moderately consistent")
            elif mean_sim > 0.70:
                p(f"  -> Somewhat consistent")
            else:
                p(f"  -> Low consistency")
        p()

    # Cross-group basis similarity
    H_htr = group_results['HTR']['H']
    H_rt13 = group_results['RT13']['H']
    cross_sim = cosine_similarity(H_htr, H_rt13)

    p("--- HTR vs RT13 Basis Cross-Similarity ---")
    header = "            " + "  ".join([f"RT13_C{k+1}" for k in range(N_COMPONENTS)])
    p(header)
    for i in range(N_COMPONENTS):
        row = f"  HTR_C{i+1}    " + "    ".join([f"{cross_sim[i,j]:.3f}" for j in range(N_COMPONENTS)])
        p(row)
    p()

    # Find best matches
    p("  Best component matches (HTR -> RT13):")
    for i in range(N_COMPONENTS):
        best_j = np.argmax(cross_sim[i])
        p(f"    HTR C{i+1} <-> RT13 C{best_j+1}: similarity = {cross_sim[i, best_j]:.3f}")
    p()

    # Overall: how much of HTR basis is shared with RT13?
    max_per_row = cross_sim.max(axis=1)
    avg_max = max_per_row.mean()
    p(f"  Average best-match similarity: {avg_max:.3f}")
    if avg_max > 0.9:
        p("  -> HTR and RT13 share very similar basis components")
    elif avg_max > 0.7:
        p("  -> HTR and RT13 share some common patterns but also have differences")
    else:
        p("  -> HTR and RT13 have substantially different basis components")
    p()

    # ── Per-trajectory NMF ──
    p("=" * 72)
    p("APPROACH B: PER-TRAJECTORY NMF")
    p("=" * 72)
    p()

    for group_name, traj_list in TRAJECTORIES.items():
        p(f"--- {group_name} Trajectories ---")
        for _, traj_name in traj_list:
            if traj_name not in per_traj_results:
                continue
            data = per_traj_results[traj_name]
            W_norm = normalize_weights(data['W'])
            avg = W_norm.mean(axis=0)

            p(f"  {traj_name} ({data['n_frames']} frames):")
            p(f"    Rel. error: {data['rel_err']:.4f} ({data['rel_err']*100:.1f}%)")
            for k in range(N_COMPONENTS):
                p(f"    C{k+1}: mean={avg[k]:.3f}, std={W_norm[:,k].std():.3f}")
            p()

        # Basis similarity within group (but components aren't aligned!)
        group_traj_names = [name for _, name in traj_list if name in per_traj_results]
        if len(group_traj_names) >= 2:
            p(f"  Basis cross-similarity within {group_name} (components NOT aligned):")
            for i in range(len(group_traj_names)):
                for j in range(i+1, len(group_traj_names)):
                    Hi = per_traj_results[group_traj_names[i]]['H']
                    Hj = per_traj_results[group_traj_names[j]]['H']
                    s = cosine_similarity(Hi, Hj)
                    max_matches = s.max(axis=1)
                    avg_match = max_matches.mean()
                    p(f"    {group_traj_names[i]} vs {group_traj_names[j]}: "
                      f"avg best-match = {avg_match:.3f}")
            p()

    # ── Summary ──
    p("=" * 72)
    p("SUMMARY: GROUP NMF vs PER-TRAJECTORY NMF")
    p("=" * 72)
    p()

    p("  Reconstruction Error Comparison:")
    p("  " + "-" * 50)
    for group_name, data in group_results.items():
        p(f"  {group_name} group NMF: {data['rel_err']:.4f}")
    for _, traj_name in [t for tl in TRAJECTORIES.values() for t in tl]:
        if traj_name in per_traj_results:
            err = per_traj_results[traj_name]['rel_err']
            p(f"  {traj_name} individual: {err:.4f}")
    p()

    p("  Key Observations:")
    p("  " + "-" * 50)
    p("  - Group NMF provides ALIGNED components across same-type trajectories,")
    p("    enabling direct comparison of evolution curves.")
    p("  - Per-trajectory NMF has lower reconstruction error (fits each trajectory")
    p("    independently) but components are NOT comparable across trajectories.")
    p("  - Group NMF within-group consistency tells us whether trajectories of the")
    p("    same type follow similar evolution patterns.")
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

    # Run both approaches
    group_results = run_group_nmf()
    per_traj_results = run_per_trajectory_nmf()

    # Generate plots
    print("\n" + "=" * 65)
    print("Generating plots...")
    print("=" * 65)

    # Approach A plots
    plot_group_basis(group_results,
                     OUTPUT_DIR / "01_group_basis_components.png")
    plot_group_evolution(group_results,
                         OUTPUT_DIR / "02_group_evolution.png")
    plot_group_stacked(group_results,
                        OUTPUT_DIR / "03_group_stacked.png")
    plot_evolution_overlay(group_results,
                           OUTPUT_DIR / "04_group_evolution_overlay.png")
    plot_basis_similarity_heatmap(group_results,
                                  OUTPUT_DIR / "05_basis_cross_similarity.png")

    # Approach B plots
    plot_per_traj_basis(per_traj_results,
                         OUTPUT_DIR / "06_per_traj_basis.png")
    plot_per_traj_evolution(per_traj_results,
                             OUTPUT_DIR / "07_per_traj_evolution.png")

    # Comparison
    plot_consistency_comparison(group_results, per_traj_results,
                                OUTPUT_DIR / "08_consistency_comparison.png")

    # Report
    print("\n" + "=" * 65)
    print("Generating report...")
    print("=" * 65)
    report = generate_report(group_results, per_traj_results,
                              OUTPUT_DIR / "REPORT.txt")
    print("\n" + report)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    return group_results, per_traj_results


if __name__ == "__main__":
    group_results, per_traj_results = main()
