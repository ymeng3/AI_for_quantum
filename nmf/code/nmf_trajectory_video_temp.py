"""
NMF Trajectory Video with Temperature

Same as nmf_trajectory_video.py but adds temperature information:
  - Large temperature readout on the original frame
  - Temperature profile plot (bottom right) with cursor

Temperature is extracted from each frame's filename.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw
from sklearn.decomposition import NMF
from pathlib import Path
from collections import OrderedDict
import imageio
import re
import io
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"
OUTPUT_DIR = Path(__file__).parent / "nmf_videos_temp"

NMF_SIZE = (128, 96)
MASK_RECT = [56, 40, 62, 50]
DISPLAY_SIZE = (320, 240)

N_COMPONENTS = 3
MAX_ITER = 1000
FPS = 15

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

COMP_COLORS = ['#e41a1c', '#377eb8', '#4daf4a']


# ─── Temperature extraction ──────────────────────────────────────────────────

def extract_temperature(filename):
    """Extract temperature from filename like '001_HL251004A_165.00C_...' or '001_RR220204A_933C_...'"""
    name = Path(filename).stem
    # Match patterns like 165.00C or 933C
    match = re.search(r'_(\d+\.?\d*)C_', name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


# ─── Preprocessing ────────────────────────────────────────────────────────────

def load_and_preprocess(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize(NMF_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    draw.rectangle(MASK_RECT, fill=0)
    return np.array(img, dtype=np.float64) / 255.0


def load_display_frame(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)
    return np.array(img)


def load_trajectory_data(traj_path):
    traj_dir = TRAJECTORY_DIR / traj_path
    if not traj_dir.exists():
        print(f"  WARNING: {traj_dir} not found")
        return None, [], []

    files = sorted(list(traj_dir.glob('*.bmp')) + list(traj_dir.glob('*.png')))
    arrays = []
    valid_files = []
    temps = []
    for f in files:
        try:
            arr = load_and_preprocess(f)
            arrays.append(arr.flatten())
            valid_files.append(f)
            temps.append(extract_temperature(f.name))
        except:
            pass

    if not arrays:
        return None, [], []
    return np.array(arrays), valid_files, temps


# ─── NMF ──────────────────────────────────────────────────────────────────────

def train_group_nmf(group_name, traj_list):
    all_rows = []
    traj_info = []

    for traj_path, traj_name in traj_list:
        V, files, temps = load_trajectory_data(traj_path)
        if V is not None:
            traj_info.append((traj_name, traj_path, V, files, temps))
            all_rows.append(V)

    if not all_rows:
        return None, []

    V_pooled = np.vstack(all_rows)
    print(f"  {group_name}: {V_pooled.shape[0]} total frames pooled")

    model = NMF(n_components=N_COMPONENTS, init='nndsvda',
                max_iter=MAX_ITER, random_state=42)
    W_pooled = model.fit_transform(V_pooled)

    offset = 0
    results = []
    for traj_name, traj_path, V, files, temps in traj_info:
        n = len(V)
        W = W_pooled[offset:offset + n]
        offset += n
        results.append({
            'name': traj_name,
            'path': traj_path,
            'W': W,
            'files': files,
            'temps': temps,
            'n_frames': n,
        })

    return model, results


# ─── Video rendering ─────────────────────────────────────────────────────────

def render_frame(fig, axes, frame_idx, n_frames, display_img, W_norm, H,
                 temps, traj_name, group_name):
    ax_orig, ax_recon, ax_bar, ax_evo, ax_temp = axes

    w = W_norm[frame_idx]
    current_temp = temps[frame_idx]

    # --- Left: Original frame with temperature overlay ---
    ax_orig.clear()
    ax_orig.imshow(display_img, cmap='gray', vmin=0, vmax=255, aspect='equal')
    ax_orig.set_title(f'Frame {frame_idx + 1}/{n_frames}', fontsize=12,
                      fontweight='bold', color='white')
    ax_orig.axis('off')

    # Temperature readout on the frame
    if current_temp is not None:
        temp_str = f'{current_temp:.0f}°C'
        ax_orig.text(0.98, 0.04, temp_str, transform=ax_orig.transAxes,
                     fontsize=18, fontweight='bold', color='#ff6644',
                     ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                               alpha=0.8, edgecolor='#ff6644'))

    # --- Right top: NMF reconstruction ---
    ax_recon.clear()
    recon = np.zeros_like(H[0])
    for k in range(N_COMPONENTS):
        recon += w[k] * H[k]
    recon_img = recon.reshape(NMF_SIZE[1], NMF_SIZE[0])
    recon_img = recon_img / (recon_img.max() + 1e-10)
    ax_recon.imshow(recon_img, cmap='gray', interpolation='nearest', aspect='equal')
    ax_recon.set_title('NMF Reconstruction', fontsize=10, color='white')
    ax_recon.axis('off')

    # --- Right middle: Stacked bar ---
    ax_bar.clear()
    left = 0
    for k in range(N_COMPONENTS):
        ax_bar.barh(0, w[k], left=left, color=COMP_COLORS[k],
                    edgecolor='none', height=0.6)
        if w[k] > 0.08:
            ax_bar.text(left + w[k] / 2, 0, f'C{k+1}\n{w[k]:.0%}',
                        ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')
        left += w[k]
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.set_title('Composition', fontsize=10, color='white')
    ax_bar.set_yticks([])
    ax_bar.tick_params(colors='white', labelsize=7)
    ax_bar.set_facecolor('#1a1a1a')

    # --- Right bottom-left: Component evolution ---
    ax_evo.clear()
    for k in range(N_COMPONENTS):
        ax_evo.plot(range(frame_idx + 1), W_norm[:frame_idx + 1, k],
                    color=COMP_COLORS[k], linewidth=1.3, alpha=0.9,
                    label=f'C{k+1}')
        if frame_idx < n_frames - 1:
            ax_evo.plot(range(frame_idx, n_frames), W_norm[frame_idx:, k],
                        color=COMP_COLORS[k], linewidth=0.4, alpha=0.12)

    ax_evo.axvline(frame_idx, color='white', linewidth=0.8, alpha=0.6,
                   linestyle='--')
    ax_evo.set_xlim(0, n_frames - 1)
    ax_evo.set_ylim(-0.02, 1.02)
    ax_evo.set_title('Component Evolution', fontsize=10, color='white')
    ax_evo.set_xlabel('Frame', fontsize=8, color='white')
    ax_evo.set_ylabel('Weight', fontsize=8, color='white')
    ax_evo.legend(loc='upper right', fontsize=7, facecolor='#2a2a2a',
                  edgecolor='gray', labelcolor='white')
    ax_evo.tick_params(colors='white', labelsize=7)
    ax_evo.set_facecolor('#1a1a1a')
    ax_evo.grid(True, alpha=0.15, color='gray')

    # --- Right bottom-right: Temperature profile ---
    ax_temp.clear()
    valid_temps = [(i, t) for i, t in enumerate(temps) if t is not None]
    if valid_temps:
        t_idx, t_vals = zip(*valid_temps)
        t_idx = list(t_idx)
        t_vals = list(t_vals)

        # Past in bold, future dimmed
        past_mask = [i for i, ti in enumerate(t_idx) if ti <= frame_idx]
        future_mask = [i for i, ti in enumerate(t_idx) if ti >= frame_idx]

        if past_mask:
            past_x = [t_idx[i] for i in past_mask]
            past_y = [t_vals[i] for i in past_mask]
            ax_temp.plot(past_x, past_y, color='#ff6644', linewidth=1.5, alpha=0.9)

        if future_mask:
            future_x = [t_idx[i] for i in future_mask]
            future_y = [t_vals[i] for i in future_mask]
            ax_temp.plot(future_x, future_y, color='#ff6644', linewidth=0.4, alpha=0.15)

        ax_temp.axvline(frame_idx, color='white', linewidth=0.8, alpha=0.6,
                        linestyle='--')

        # Current temp marker
        if current_temp is not None:
            ax_temp.plot(frame_idx, current_temp, 'o', color='#ff6644',
                         markersize=6, zorder=5)
            ax_temp.text(frame_idx, current_temp,
                         f'  {current_temp:.0f}°C', fontsize=8,
                         color='#ff6644', va='center', fontweight='bold')

    ax_temp.set_xlim(0, n_frames - 1)
    ax_temp.set_title('Temperature', fontsize=10, color='white')
    ax_temp.set_xlabel('Frame', fontsize=8, color='white')
    ax_temp.set_ylabel('°C', fontsize=8, color='white')
    ax_temp.tick_params(colors='white', labelsize=7)
    ax_temp.set_facecolor('#1a1a1a')
    ax_temp.grid(True, alpha=0.15, color='gray')


def fig_to_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.1, dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    arr = np.array(img)
    buf.close()
    return arr


def normalize_weights(W):
    return W / (W.sum(axis=1, keepdims=True) + 1e-10)


def create_video(model, traj_data, group_name, output_path, fps=FPS,
                 target_duration=30):
    W = traj_data['W']
    files = traj_data['files']
    temps = traj_data['temps']
    n_frames = traj_data['n_frames']
    traj_name = traj_data['name']
    H = model.components_

    W_norm = normalize_weights(W)

    H_display = H.copy()
    for k in range(N_COMPONENTS):
        H_display[k] = H_display[k] / (H_display[k].max() + 1e-10)

    frame_skip = max(1, n_frames // (target_duration * fps))

    # Layout: 2 columns, right column has 4 rows
    fig = plt.figure(figsize=(14, 6), facecolor='#111111')
    gs = GridSpec(4, 2, figure=fig, width_ratios=[1.2, 1],
                  height_ratios=[3, 1, 3, 3], hspace=0.5, wspace=0.3)

    ax_orig = fig.add_subplot(gs[:, 0])    # left: original frame
    ax_recon = fig.add_subplot(gs[0, 1])   # right top: reconstruction
    ax_bar = fig.add_subplot(gs[1, 1])     # right: stacked bar
    ax_evo = fig.add_subplot(gs[2, 1])     # right: component evolution
    ax_temp = fig.add_subplot(gs[3, 1])    # right bottom: temperature

    axes = (ax_orig, ax_recon, ax_bar, ax_evo, ax_temp)

    fig.suptitle(f'{traj_name} ({group_name}) — NMF Decomposition + Temperature',
                 fontsize=13, fontweight='bold', color='white', y=0.99)

    frame_indices = list(range(0, n_frames, frame_skip))
    total_video_frames = len(frame_indices)

    print(f"  Rendering {total_video_frames} video frames "
          f"(skip={frame_skip}, {n_frames} total)...")

    # Get dimensions from first frame
    display_img = load_display_frame(files[0])
    render_frame(fig, axes, 0, n_frames, display_img, W_norm, H_display,
                 temps, traj_name, group_name)
    first_arr = fig_to_array(fig)
    h, w_px, _ = first_arr.shape

    frames = []
    for progress_idx, fi in enumerate(frame_indices):
        display_img = load_display_frame(files[fi])
        render_frame(fig, axes, fi, n_frames, display_img, W_norm, H_display,
                     temps, traj_name, group_name)
        arr = fig_to_array(fig)

        if arr.shape[0] != h or arr.shape[1] != w_px:
            padded = np.zeros((h, w_px, 3), dtype=np.uint8)
            min_h = min(arr.shape[0], h)
            min_w = min(arr.shape[1], w_px)
            padded[:min_h, :min_w] = arr[:min_h, :min_w]
            arr = padded

        if arr.shape[0] % 2 != 0:
            arr = arr[:-1, :, :]
        if arr.shape[1] % 2 != 0:
            arr = arr[:, :-1, :]

        frames.append(arr)

        if (progress_idx + 1) % 100 == 0 or progress_idx == 0:
            pct = (progress_idx + 1) / total_video_frames * 100
            print(f"    {progress_idx + 1}/{total_video_frames} ({pct:.0f}%)")

    plt.close(fig)

    imageio.mimwrite(str(output_path), frames, fps=fps, codec='libx264',
                     output_params=['-pix_fmt', 'yuv420p'])

    # Temperature range summary
    valid_temps = [t for t in temps if t is not None]
    temp_range = f"{min(valid_temps):.0f}-{max(valid_temps):.0f}°C" if valid_temps else "N/A"

    print(f"  Saved: {output_path.name} ({total_video_frames} frames, {fps} fps, "
          f"{total_video_frames / fps:.1f}s, temp: {temp_range})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Training group NMF models...")
    group_models = {}
    all_traj = []

    for group_name, traj_list in TRAJECTORIES.items():
        model, results = train_group_nmf(group_name, traj_list)
        if model is not None:
            group_models[group_name] = model
            for r in results:
                r['group'] = group_name
                all_traj.append(r)

    print(f"\nCreating videos for {len(all_traj)} trajectories...")
    for traj in all_traj:
        group = traj['group']
        model = group_models[group]

        output_path = OUTPUT_DIR / f"{traj['name']}_{group}_temp.mp4"
        print(f"\n--- {traj['name']} ({group}, {traj['n_frames']} frames) ---")
        create_video(model, traj, group, output_path)

    print(f"\nAll videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
