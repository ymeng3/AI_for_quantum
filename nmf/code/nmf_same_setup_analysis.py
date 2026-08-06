"""
NMF Same-Setup Analysis

Focused analysis on same-setup trajectory pairs:
  - 2025 HTR: HL251004A and HL251004B (same RHEED system)
  - 2022 RT13: RR220204A and RR220411A (same RHEED system)

Questions:
  1. Do decomposition components look similar / can we identify them?
  2. Are component evolution patterns consistent across temperature and phase changes?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from collections import OrderedDict
import re
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"
OUTPUT_DIR = Path(__file__).parent / "nmf_same_setup_results"

IMG_SIZE = (128, 96)
MASK_RECT = [56, 40, 62, 50]
N_COMPONENTS = 3
MAX_ITER = 1000

# Same-setup pairs only
PAIRS = OrderedDict({
    'HTR (2025 setup)': [
        ('2025-10-04/A', 'HL251004A'),
        ('2025-10-04/B', 'HL251004B'),
    ],
    'RT13 (2022 setup)': [
        ('2022-02-04', 'RR220204A'),
        ('2022-04-11', 'RR220411A'),
    ]
})

COLORS = ['#e41a1c', '#377eb8', '#4daf4a']


# ─── Temperature extraction ──────────────────────────────────────────────────

def extract_temperature(filename):
    """Extract temperature in Celsius from filename."""
    name = filename.stem if hasattr(filename, 'stem') else str(filename)

    # Pattern 1: "165.00C" or "914.70C" (2025 numbered)
    m = re.search(r'_(\d+\.?\d*)C', name)
    if m:
        return float(m.group(1))

    # Pattern 2: "933C" (2022)
    m = re.search(r'_(\d+)C_', name)
    if m:
        return float(m.group(1))

    return None


# ─── Preprocessing ────────────────────────────────────────────────────────────

def load_and_preprocess(img_path):
    """Load → grayscale → resize → mask → CLAHE-like normalize."""
    img = Image.open(img_path).convert('L')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    draw.rectangle(MASK_RECT, fill=0)

    # Percentile normalization (best for same-setup)
    arr = np.array(img, dtype=np.float64)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 > p2:
        arr = np.clip(arr, p2, p98)
        arr = (arr - p2) / (p98 - p2)
    else:
        arr = arr / 255.0
    return arr


def load_trajectory_with_temps(traj_path):
    """Load trajectory, extract temperatures, return (matrix, temps, files)."""
    traj_dir = TRAJECTORY_DIR / traj_path
    if not traj_dir.exists():
        print(f"  WARNING: {traj_dir} not found")
        return None, [], []

    files = sorted(list(traj_dir.glob('*.bmp')) + list(traj_dir.glob('*.png')))
    arrays = []
    temps = []
    valid_files = []

    for f in files:
        try:
            arr = load_and_preprocess(f)
            t = extract_temperature(f)
            arrays.append(arr.flatten())
            temps.append(t)
            valid_files.append(f)
        except:
            pass

    if not arrays:
        return None, [], []
    return np.array(arrays), temps, valid_files


# ─── Analysis ─────────────────────────────────────────────────────────────────

def run_pair_nmf(pair_name, traj_list):
    """Pool two trajectories, train NMF, return results."""
    print(f"\n{'='*60}")
    print(f"  {pair_name}")
    print(f"{'='*60}")

    traj_data = []
    all_rows = []

    for traj_path, traj_name in traj_list:
        V, temps, files = load_trajectory_with_temps(traj_path)
        if V is not None:
            traj_data.append({
                'name': traj_name,
                'V': V,
                'temps': temps,
                'n_frames': len(V),
            })
            all_rows.append(V)
            valid_temps = [t for t in temps if t is not None]
            if valid_temps:
                print(f"  {traj_name}: {len(V)} frames, "
                      f"temp range: {min(valid_temps):.0f}-{max(valid_temps):.0f}C")
            else:
                print(f"  {traj_name}: {len(V)} frames, no temp data")

    V_pooled = np.vstack(all_rows)
    print(f"  Pooled: {V_pooled.shape[0]} frames")

    # Train NMF
    model = NMF(n_components=N_COMPONENTS, init='nndsvda',
                max_iter=MAX_ITER, random_state=42)
    W_pooled = model.fit_transform(V_pooled)
    H = model.components_

    V_recon = W_pooled @ H
    rel_err = np.linalg.norm(V_pooled - V_recon) / (np.linalg.norm(V_pooled) + 1e-10)
    print(f"  Rel. error: {rel_err:.4f} ({rel_err*100:.1f}%), iter: {model.n_iter_}")

    # Split W per trajectory
    offset = 0
    for td in traj_data:
        n = td['n_frames']
        td['W'] = W_pooled[offset:offset + n]
        td['W_norm'] = td['W'] / (td['W'].sum(axis=1, keepdims=True) + 1e-10)
        td['avg'] = td['W_norm'].mean(axis=0)
        offset += n

    # Consistency
    if len(traj_data) == 2:
        sim = cosine_similarity(
            traj_data[0]['avg'].reshape(1, -1),
            traj_data[1]['avg'].reshape(1, -1)
        )[0, 0]
        print(f"  Consistency ({traj_data[0]['name']} vs {traj_data[1]['name']}): {sim:.3f}")

    return {
        'H': H,
        'rel_err': rel_err,
        'trajectories': traj_data,
        'model': model,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_basis_components(results, pair_name, output_path):
    """Large view of basis components with annotations."""
    H = results['H']
    fig, axes = plt.subplots(1, N_COMPONENTS, figsize=(6 * N_COMPONENTS, 5))

    for k in range(N_COMPONENTS):
        ax = axes[k]
        comp = H[k].reshape(IMG_SIZE[1], IMG_SIZE[0])
        comp = comp / (comp.max() + 1e-10)
        ax.imshow(comp, cmap='hot', interpolation='nearest')
        ax.set_title(f'Component {k+1}', fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f'{pair_name} — NMF Basis Components\n'
                 f'({N_COMPONENTS} components, err={results["rel_err"]:.1%})',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_evolution_vs_frame(results, pair_name, output_path):
    """Side-by-side: component weights over frame number."""
    trajs = results['trajectories']
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    for col, td in enumerate(trajs):
        ax = axes[col]
        W_norm = td['W_norm']
        for k in range(N_COMPONENTS):
            ax.plot(W_norm[:, k], color=COLORS[k],
                    label=f'C{k+1}', linewidth=1.3, alpha=0.85)
        ax.set_title(f"{td['name']} ({td['n_frames']} frames)",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Normalized Weight', fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{pair_name} — Component Evolution (by frame)', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_evolution_vs_temperature(results, pair_name, output_path):
    """Side-by-side: component weights vs temperature."""
    trajs = results['trajectories']
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    for col, td in enumerate(trajs):
        ax = axes[col]
        temps = td['temps']
        W_norm = td['W_norm']

        # Filter frames with valid temperature
        valid = [(i, t) for i, t in enumerate(temps) if t is not None]
        if not valid:
            ax.text(0.5, 0.5, 'No temperature data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        indices = [v[0] for v in valid]
        temp_vals = [v[1] for v in valid]

        for k in range(N_COMPONENTS):
            vals = W_norm[indices, k]
            ax.plot(temp_vals, vals, color=COLORS[k],
                    label=f'C{k+1}', linewidth=1.0, alpha=0.7,
                    markersize=1)

        ax.set_title(f"{td['name']} ({td['n_frames']} frames)",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Temperature (C)', fontsize=12)
        ax.set_ylabel('Normalized Weight', fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{pair_name} — Component Evolution (by temperature)', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_evolution_overlay_normalized_time(results, pair_name, output_path):
    """Overlay both trajectories on normalized [0,1] time axis per component."""
    trajs = results['trajectories']
    fig, axes = plt.subplots(1, N_COMPONENTS, figsize=(7 * N_COMPONENTS, 5))

    line_styles = ['-', '--']
    traj_colors = ['#e41a1c', '#377eb8']

    for k in range(N_COMPONENTS):
        ax = axes[k]
        for t_idx, td in enumerate(trajs):
            W_norm = td['W_norm']
            x = np.linspace(0, 1, len(W_norm))
            ax.plot(x, W_norm[:, k],
                    color=traj_colors[t_idx],
                    linestyle=line_styles[t_idx],
                    label=td['name'], linewidth=1.5, alpha=0.8)
        ax.set_title(f'Component {k+1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Normalized Time (0→1)', fontsize=11)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{pair_name} — Evolution Overlay (normalized time)\n'
                 'Similar shapes = consistent pattern', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_stacked_comparison(results, pair_name, output_path):
    """Stacked area plots side by side."""
    trajs = results['trajectories']
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for col, td in enumerate(trajs):
        ax = axes[col]
        W_norm = td['W_norm']
        ax.stackplot(range(len(W_norm)),
                     [W_norm[:, k] for k in range(N_COMPONENTS)],
                     labels=[f'C{k+1}' for k in range(N_COMPONENTS)],
                     colors=COLORS, alpha=0.75)
        ax.set_title(f"{td['name']}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Composition', fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=10)

    plt.suptitle(f'{pair_name} — Composition (Stacked)', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_transition_analysis(results, pair_name, output_path):
    """Identify phase transitions: where dominant component changes."""
    trajs = results['trajectories']
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    for col, td in enumerate(trajs):
        W_norm = td['W_norm']
        temps = td['temps']
        dominant = np.argmax(W_norm, axis=1)

        # Top row: dominant component colored by phase
        ax = axes[0, col]
        for k in range(N_COMPONENTS):
            mask = dominant == k
            frames = np.where(mask)[0]
            if len(frames) > 0:
                ax.scatter(frames, W_norm[frames, k], c=COLORS[k],
                           s=3, alpha=0.6, label=f'C{k+1} dominant')
        ax.set_title(f"{td['name']} — Dominant Component", fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Weight of Dominant Component')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, markerscale=3)
        ax.grid(True, alpha=0.3)

        # Bottom row: phase timeline (color bar)
        ax = axes[1, col]
        # Create a color-coded timeline
        phase_colors = np.array([COLORS[d] for d in dominant])

        # Plot as colored segments
        for i in range(len(dominant)):
            ax.axvspan(i, i+1, color=COLORS[dominant[i]], alpha=0.7)

        # Overlay temperature if available
        valid_temps = [(i, t) for i, t in enumerate(temps) if t is not None]
        if valid_temps:
            t_frames = [v[0] for v in valid_temps]
            t_vals = [v[1] for v in valid_temps]
            ax2 = ax.twinx()
            ax2.plot(t_frames, t_vals, 'k-', linewidth=1.5, alpha=0.8)
            ax2.set_ylabel('Temperature (C)', fontsize=11)

        ax.set_title(f"{td['name']} — Phase Timeline + Temperature", fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_yticks([])
        ax.set_ylabel('Phase')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=COLORS[k], alpha=0.7, label=f'C{k+1}')
                           for k in range(N_COMPONENTS)]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.suptitle(f'{pair_name} — Phase Transition Analysis', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_transition_temperatures(results, pair_name, output_path):
    """Find and compare temperatures where phase transitions occur."""
    trajs = results['trajectories']
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for col, td in enumerate(trajs):
        ax = axes[col]
        W_norm = td['W_norm']
        temps = td['temps']
        dominant = np.argmax(W_norm, axis=1)

        # Find transition points (where dominant component changes)
        transitions = []
        for i in range(1, len(dominant)):
            if dominant[i] != dominant[i-1]:
                t = temps[i] if temps[i] is not None else None
                transitions.append({
                    'frame': i,
                    'temp': t,
                    'from': dominant[i-1],
                    'to': dominant[i],
                })

        # Plot W_norm vs temperature
        valid = [(i, t) for i, t in enumerate(temps) if t is not None]
        if valid:
            indices = [v[0] for v in valid]
            temp_vals = [v[1] for v in valid]

            for k in range(N_COMPONENTS):
                vals = W_norm[indices, k]
                ax.plot(temp_vals, vals, color=COLORS[k],
                        label=f'C{k+1}', linewidth=1.2, alpha=0.8)

            # Mark major transitions (sustained, not just flicker)
            # Find sustained transitions: dominant for at least 10 frames
            sustained = []
            i = 0
            segments = []
            while i < len(dominant):
                j = i
                while j < len(dominant) and dominant[j] == dominant[i]:
                    j += 1
                segments.append((i, j, dominant[i]))
                i = j

            for seg_idx in range(1, len(segments)):
                start_prev, end_prev, comp_prev = segments[seg_idx - 1]
                start_cur, end_cur, comp_cur = segments[seg_idx]
                dur_prev = end_prev - start_prev
                dur_cur = end_cur - start_cur
                if dur_prev >= 10 and dur_cur >= 10:
                    t = temps[start_cur] if temps[start_cur] is not None else None
                    if t is not None:
                        sustained.append({
                            'temp': t,
                            'from': comp_prev,
                            'to': comp_cur,
                            'frame': start_cur,
                        })
                        ax.axvline(t, color='gray', linestyle='--', alpha=0.5)
                        ax.annotate(f'C{comp_prev+1}→C{comp_cur+1}\n{t:.0f}C',
                                   xy=(t, 0.95), fontsize=8,
                                   ha='center', va='top',
                                   backgroundcolor='white')

        ax.set_title(f"{td['name']}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Temperature (C)', fontsize=12)
        ax.set_ylabel('Normalized Weight', fontsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{pair_name} — Component Weights vs Temperature\n'
                 'Dashed lines = sustained phase transitions', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_avg_composition_comparison(all_results, output_path):
    """Bar chart: average composition of each trajectory across all pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for p_idx, (pair_name, results) in enumerate(all_results.items()):
        ax = axes[p_idx]
        trajs = results['trajectories']
        names = [t['name'] for t in trajs]
        avgs = [t['avg'] for t in trajs]

        x = np.arange(len(names))
        bottom = np.zeros(len(names))
        for k in range(N_COMPONENTS):
            vals = [a[k] for a in avgs]
            ax.bar(x, vals, bottom=bottom, label=f'C{k+1}',
                   color=COLORS[k], alpha=0.85)
            # Add percentage labels
            for i, v in enumerate(vals):
                if v > 0.05:
                    ax.text(i, bottom[i] + v/2, f'{v:.0%}',
                            ha='center', va='center', fontsize=11,
                            fontweight='bold', color='white')
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=12)
        ax.set_ylabel('Average Composition', fontsize=12)
        ax.set_title(pair_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)

        # Cosine similarity annotation
        sim = cosine_similarity(
            avgs[0].reshape(1, -1), avgs[1].reshape(1, -1)
        )[0, 0]
        ax.text(0.5, -0.12, f'Cosine similarity: {sim:.3f}',
                transform=ax.transAxes, ha='center', fontsize=12,
                fontweight='bold')

    plt.suptitle('Average Composition Comparison (same-setup pairs)', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Report ───────────────────────────────────────────────────────────────────

def generate_report(all_results, report_path):
    """Detailed text report."""
    lines = []
    def p(s=''):
        lines.append(s)

    p("=" * 72)
    p("NMF SAME-SETUP ANALYSIS REPORT")
    p("=" * 72)
    p()
    p(f"Image: {IMG_SIZE[0]}x{IMG_SIZE[1]} grayscale, percentile normalized")
    p(f"Components: {N_COMPONENTS}")
    p()

    for pair_name, results in all_results.items():
        p("=" * 72)
        p(f"  {pair_name}")
        p("=" * 72)
        p()
        p(f"  Reconstruction error: {results['rel_err']:.4f} ({results['rel_err']*100:.1f}%)")
        p()

        # Component descriptions
        p("  1. BASIS COMPONENTS")
        p("  " + "-" * 40)
        H = results['H']
        for k in range(N_COMPONENTS):
            comp = H[k].reshape(IMG_SIZE[1], IMG_SIZE[0])
            # Basic stats about where intensity is concentrated
            total = comp.sum()
            top_half = comp[:IMG_SIZE[1]//2, :].sum() / total
            bottom_half = comp[IMG_SIZE[1]//2:, :].sum() / total
            center = comp[IMG_SIZE[1]//4:3*IMG_SIZE[1]//4,
                         IMG_SIZE[0]//4:3*IMG_SIZE[0]//4].sum() / total
            p(f"  C{k+1}: top={top_half:.0%}, bottom={bottom_half:.0%}, "
              f"center={center:.0%}, max={comp.max():.4f}")
        p()

        # Per-trajectory analysis
        p("  2. PER-TRAJECTORY COMPOSITION")
        p("  " + "-" * 40)
        trajs = results['trajectories']
        for td in trajs:
            W_norm = td['W_norm']
            avg = td['avg']
            p(f"  {td['name']} ({td['n_frames']} frames):")
            for k in range(N_COMPONENTS):
                p(f"    C{k+1}: mean={avg[k]:.3f}, std={W_norm[:,k].std():.3f}, "
                  f"range=[{W_norm[:,k].min():.3f}, {W_norm[:,k].max():.3f}]")
            dominant = np.argmax(avg)
            p(f"    Overall dominant: C{dominant+1} ({avg[dominant]:.1%})")
            p()

        # Consistency
        p("  3. CONSISTENCY")
        p("  " + "-" * 40)
        if len(trajs) == 2:
            sim = cosine_similarity(
                trajs[0]['avg'].reshape(1, -1),
                trajs[1]['avg'].reshape(1, -1)
            )[0, 0]
            p(f"  Average composition cosine similarity: {sim:.3f}")

            # Per-component correlation
            for k in range(N_COMPONENTS):
                # Resample both to same length for correlation
                w1 = trajs[0]['W_norm'][:, k]
                w2 = trajs[1]['W_norm'][:, k]
                n = min(len(w1), len(w2))
                # Resample to common length
                x1 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(w1)), w1)
                x2 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(w2)), w2)
                corr = np.corrcoef(x1, x2)[0, 1]
                p(f"  C{k+1} evolution correlation (resampled): {corr:.3f}")
            p()

        # Phase transitions
        p("  4. PHASE TRANSITIONS")
        p("  " + "-" * 40)
        for td in trajs:
            W_norm = td['W_norm']
            temps = td['temps']
            dominant = np.argmax(W_norm, axis=1)

            # Find sustained segments
            segments = []
            i = 0
            while i < len(dominant):
                j = i
                while j < len(dominant) and dominant[j] == dominant[i]:
                    j += 1
                segments.append((i, j, dominant[i], j - i))
                i = j

            p(f"  {td['name']}:")
            p(f"    Total segments: {len(segments)}")

            # Report sustained segments (>= 10 frames)
            sustained = [(s, e, c, d) for s, e, c, d in segments if d >= 10]
            p(f"    Sustained (>=10 frames): {len(sustained)}")
            for s, e, c, dur in sustained:
                t_start = temps[s] if temps[s] is not None else '?'
                t_end = temps[e-1] if temps[e-1] is not None else '?'
                if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)):
                    p(f"      Frames {s:4d}-{e:4d} ({dur:4d} frames): C{c+1} "
                      f"@ {t_start:.0f}-{t_end:.0f}C")
                else:
                    p(f"      Frames {s:4d}-{e:4d} ({dur:4d} frames): C{c+1}")

            # Major transitions between sustained segments
            p(f"    Major transitions:")
            for seg_idx in range(1, len(sustained)):
                s_prev, e_prev, c_prev, d_prev = sustained[seg_idx - 1]
                s_cur, e_cur, c_cur, d_cur = sustained[seg_idx]
                t = temps[s_cur] if temps[s_cur] is not None else None
                if t is not None:
                    p(f"      C{c_prev+1} -> C{c_cur+1} at frame {s_cur} ({t:.0f}C)")
                else:
                    p(f"      C{c_prev+1} -> C{c_cur+1} at frame {s_cur}")
            p()

        # Cross-trajectory transition comparison
        if len(trajs) == 2:
            p("  5. TRANSITION COMPARISON")
            p("  " + "-" * 40)
            for t_idx, td in enumerate(trajs):
                dominant = np.argmax(td['W_norm'], axis=1)
                segments = []
                i = 0
                while i < len(dominant):
                    j = i
                    while j < len(dominant) and dominant[j] == dominant[i]:
                        j += 1
                    if j - i >= 10:
                        segments.append((dominant[i], j - i))
                    i = j
                phase_sequence = [f'C{c+1}' for c, d in segments]
                p(f"  {td['name']} phase sequence: {' -> '.join(phase_sequence)}")
            p()

    p("=" * 72)
    p("END OF REPORT")
    p("=" * 72)

    report_text = '\n'.join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    return report_text


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_results = OrderedDict()

    for pair_name, traj_list in PAIRS.items():
        results = run_pair_nmf(pair_name, traj_list)
        all_results[pair_name] = results

    # Generate plots per pair
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")

    for p_idx, (pair_name, results) in enumerate(all_results.items()):
        prefix = f"{'htr' if 'HTR' in pair_name else 'rt13'}"
        print(f"\n  --- {pair_name} ---")

        plot_basis_components(results, pair_name,
                               OUTPUT_DIR / f"{prefix}_01_basis.png")
        plot_evolution_vs_frame(results, pair_name,
                                 OUTPUT_DIR / f"{prefix}_02_evolution_frame.png")
        plot_evolution_vs_temperature(results, pair_name,
                                       OUTPUT_DIR / f"{prefix}_03_evolution_temp.png")
        plot_evolution_overlay_normalized_time(results, pair_name,
                                                OUTPUT_DIR / f"{prefix}_04_overlay.png")
        plot_stacked_comparison(results, pair_name,
                                 OUTPUT_DIR / f"{prefix}_05_stacked.png")
        plot_phase_transition_analysis(results, pair_name,
                                        OUTPUT_DIR / f"{prefix}_06_phases.png")
        plot_transition_temperatures(results, pair_name,
                                      OUTPUT_DIR / f"{prefix}_07_transitions_temp.png")

        for fn in sorted(OUTPUT_DIR.glob(f'{prefix}_*.png')):
            print(f"    Saved: {fn.name}")

    # Cross-pair comparison
    plot_avg_composition_comparison(all_results,
                                     OUTPUT_DIR / "00_avg_composition.png")
    print(f"    Saved: 00_avg_composition.png")

    # Report
    print(f"\n{'='*60}")
    print("Generating report...")
    print(f"{'='*60}")
    report = generate_report(all_results, OUTPUT_DIR / "REPORT.txt")
    print("\n" + report)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    return all_results


if __name__ == "__main__":
    all_results = main()
