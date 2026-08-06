"""
NMF Trajectory Video

Creates a video for each trajectory showing:
  Left:         Original RHEED frame
  Right top:    NMF reconstruction (sum of weighted components)
  Right middle: Stacked bar of current frame's component weights
  Right bottom: Component evolution over time (line plot with cursor)

Uses separate-group NMF (shared basis per HTR/RT13 group).
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
import imageio.v3 as iio
import io
import sys
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"
OUTPUT_DIR = Path(__file__).parent / "nmf_videos"

# NMF image size (for decomposition)
NMF_SIZE = (128, 96)
MASK_RECT = [56, 40, 62, 50]

# Display size for the original frame (larger, for visibility)
DISPLAY_SIZE = (320, 240)

N_COMPONENTS = 3
MAX_ITER = 1000
FPS = 15
FRAME_SKIP = 1  # Use every Nth frame (1 = all frames)

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
COMP_COLORS_RGBA = [(0.89, 0.10, 0.11), (0.22, 0.49, 0.72), (0.30, 0.69, 0.29)]


# ─── Preprocessing ────────────────────────────────────────────────────────────

def load_and_preprocess(img_path):
    """Load → grayscale → resize → mask → normalize."""
    img = Image.open(img_path).convert('L')
    img = img.resize(NMF_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    draw.rectangle(MASK_RECT, fill=0)
    return np.array(img, dtype=np.float64) / 255.0


def load_display_frame(img_path):
    """Load frame at display resolution for the video."""
    img = Image.open(img_path).convert('L')
    img = img.resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)
    return np.array(img)


def load_trajectory_data(traj_path):
    """Load all frames, return NMF matrix and file list."""
    traj_dir = TRAJECTORY_DIR / traj_path
    if not traj_dir.exists():
        print(f"  WARNING: {traj_dir} not found")
        return None, []

    files = sorted(list(traj_dir.glob('*.bmp')) + list(traj_dir.glob('*.png')))
    arrays = []
    valid_files = []
    for f in files:
        try:
            arr = load_and_preprocess(f)
            arrays.append(arr.flatten())
            valid_files.append(f)
        except:
            pass

    if not arrays:
        return None, []
    return np.array(arrays), valid_files


# ─── NMF ──────────────────────────────────────────────────────────────────────

def train_group_nmf(group_name, traj_list):
    """Pool trajectories in a group, train NMF, return model + per-traj data."""
    all_rows = []
    traj_info = []

    for traj_path, traj_name in traj_list:
        V, files = load_trajectory_data(traj_path)
        if V is not None:
            traj_info.append((traj_name, traj_path, V, files))
            all_rows.append(V)

    if not all_rows:
        return None, []

    V_pooled = np.vstack(all_rows)
    print(f"  {group_name}: {V_pooled.shape[0]} total frames pooled")

    model = NMF(n_components=N_COMPONENTS, init='nndsvda',
                max_iter=MAX_ITER, random_state=42)
    W_pooled = model.fit_transform(V_pooled)
    H = model.components_

    # Split W back per trajectory
    offset = 0
    results = []
    for traj_name, traj_path, V, files in traj_info:
        n = len(V)
        W = W_pooled[offset:offset + n]
        offset += n
        results.append({
            'name': traj_name,
            'path': traj_path,
            'W': W,
            'files': files,
            'n_frames': n,
        })

    return model, results


# ─── Video rendering ─────────────────────────────────────────────────────────

def render_frame(fig, axes, frame_idx, n_frames, display_img, W_norm, H,
                 traj_name, group_name):
    """Render a single video frame onto the figure."""
    ax_orig, ax_recon, ax_bar, ax_line = axes

    # --- Left: Original frame ---
    ax_orig.clear()
    ax_orig.imshow(display_img, cmap='gray', vmin=0, vmax=255, aspect='equal')
    ax_orig.set_title(f'Frame {frame_idx + 1}/{n_frames}', fontsize=13,
                      fontweight='bold', color='white')
    ax_orig.axis('off')

    # --- Right top: NMF reconstruction ---
    ax_recon.clear()
    w = W_norm[frame_idx]
    # Build weighted reconstruction
    recon = np.zeros_like(H[0])
    for k in range(N_COMPONENTS):
        recon += w[k] * H[k]
    recon_img = recon.reshape(NMF_SIZE[1], NMF_SIZE[0])
    recon_img = recon_img / (recon_img.max() + 1e-10)
    ax_recon.imshow(recon_img, cmap='gray', interpolation='nearest', aspect='equal')
    ax_recon.set_title('NMF Reconstruction', fontsize=11, color='white')
    ax_recon.axis('off')

    # --- Right middle: Stacked bar for current frame ---
    ax_bar.clear()
    left = 0
    for k in range(N_COMPONENTS):
        ax_bar.barh(0, w[k], left=left, color=COMP_COLORS[k],
                    edgecolor='none', height=0.6)
        if w[k] > 0.08:
            ax_bar.text(left + w[k] / 2, 0, f'C{k+1}\n{w[k]:.0%}',
                        ha='center', va='center', fontsize=9,
                        fontweight='bold', color='white')
        left += w[k]
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.set_title('Current Composition', fontsize=11, color='white')
    ax_bar.set_xlabel('Weight', fontsize=9, color='white')
    ax_bar.set_yticks([])
    ax_bar.tick_params(colors='white', labelsize=8)
    ax_bar.set_facecolor('#1a1a1a')

    # --- Right bottom: Evolution line plot ---
    ax_line.clear()
    x_range = np.arange(n_frames)
    for k in range(N_COMPONENTS):
        ax_line.plot(x_range[:frame_idx + 1], W_norm[:frame_idx + 1, k],
                     color=COMP_COLORS[k], linewidth=1.5, alpha=0.9,
                     label=f'C{k+1}')
        # Dim future
        if frame_idx < n_frames - 1:
            ax_line.plot(x_range[frame_idx:], W_norm[frame_idx:, k],
                         color=COMP_COLORS[k], linewidth=0.5, alpha=0.15)

    # Cursor line
    ax_line.axvline(frame_idx, color='white', linewidth=1, alpha=0.7,
                    linestyle='--')
    ax_line.set_xlim(0, n_frames - 1)
    ax_line.set_ylim(-0.02, 1.02)
    ax_line.set_title('Component Evolution', fontsize=11, color='white')
    ax_line.set_xlabel('Frame', fontsize=9, color='white')
    ax_line.set_ylabel('Weight', fontsize=9, color='white')
    ax_line.legend(loc='upper right', fontsize=8, facecolor='#2a2a2a',
                   edgecolor='gray', labelcolor='white')
    ax_line.tick_params(colors='white', labelsize=8)
    ax_line.set_facecolor('#1a1a1a')
    ax_line.grid(True, alpha=0.2, color='gray')


def fig_to_array(fig):
    """Convert matplotlib figure to numpy RGB array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.1, dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    arr = np.array(img)
    buf.close()
    return arr


def create_video(model, traj_data, group_name, output_path, fps=FPS,
                 frame_skip=FRAME_SKIP):
    """Create MP4 video for a single trajectory."""
    W = traj_data['W']
    files = traj_data['files']
    n_frames = traj_data['n_frames']
    traj_name = traj_data['name']
    H = model.components_

    # Normalize W
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)

    # Normalize H for reconstruction display
    H_display = H.copy()
    for k in range(N_COMPONENTS):
        H_display[k] = H_display[k] / (H_display[k].max() + 1e-10)

    # Set up figure with dark background
    fig = plt.figure(figsize=(12, 5), facecolor='#111111')
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.3, 1],
                  height_ratios=[3, 1, 3], hspace=0.4, wspace=0.3)

    ax_orig = fig.add_subplot(gs[:, 0])   # left: original frame (tall)
    ax_recon = fig.add_subplot(gs[0, 1])  # right top: reconstruction
    ax_bar = fig.add_subplot(gs[1, 1])    # right middle: stacked bar
    ax_line = fig.add_subplot(gs[2, 1])   # right bottom: evolution

    axes = (ax_orig, ax_recon, ax_bar, ax_line)

    # Title
    fig.suptitle(f'{traj_name} ({group_name}) — NMF Decomposition',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    # Select frames to render
    frame_indices = list(range(0, n_frames, frame_skip))
    total_video_frames = len(frame_indices)

    print(f"  Rendering {total_video_frames} video frames "
          f"(skip={frame_skip}, {n_frames} total)...")

    # Render first frame to get dimensions
    display_img = load_display_frame(files[0])
    render_frame(fig, axes, 0, n_frames, display_img, W_norm, H_display,
                 traj_name, group_name)
    first_arr = fig_to_array(fig)
    h, w, _ = first_arr.shape

    # Collect all frames in memory, then write at once
    frames = []

    for progress_idx, fi in enumerate(frame_indices):
        display_img = load_display_frame(files[fi])
        render_frame(fig, axes, fi, n_frames, display_img, W_norm, H_display,
                     traj_name, group_name)
        arr = fig_to_array(fig)

        # Ensure consistent dimensions
        if arr.shape[0] != h or arr.shape[1] != w:
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            min_h = min(arr.shape[0], h)
            min_w = min(arr.shape[1], w)
            padded[:min_h, :min_w] = arr[:min_h, :min_w]
            arr = padded

        # Ensure even dimensions for h264
        if arr.shape[0] % 2 != 0:
            arr = arr[:-1, :, :]
        if arr.shape[1] % 2 != 0:
            arr = arr[:, :-1, :]

        frames.append(arr)

        if (progress_idx + 1) % 100 == 0 or progress_idx == 0:
            pct = (progress_idx + 1) / total_video_frames * 100
            print(f"    {progress_idx + 1}/{total_video_frames} ({pct:.0f}%)")

    plt.close(fig)

    # Write video with imageio
    import imageio
    imageio.mimwrite(str(output_path), frames, fps=fps, codec='libx264',
                     output_params=['-pix_fmt', 'yuv420p'])
    print(f"  Saved: {output_path.name} ({total_video_frames} frames, {fps} fps, "
          f"{total_video_frames / fps:.1f}s)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Determine frame skip per trajectory to keep videos reasonable length
    # Target ~30 seconds per video
    target_duration = 30  # seconds

    # Train group NMF
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

    # Create videos
    print(f"\nCreating videos for {len(all_traj)} trajectories...")
    for traj in all_traj:
        group = traj['group']
        model = group_models[group]
        n_frames = traj['n_frames']

        # Calculate frame skip for ~target_duration video
        skip = max(1, n_frames // (target_duration * FPS))

        output_path = OUTPUT_DIR / f"{traj['name']}_{group}.mp4"
        print(f"\n--- {traj['name']} ({group}, {n_frames} frames, skip={skip}) ---")
        create_video(model, traj, group, output_path, fps=FPS, frame_skip=skip)

    print(f"\nAll videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
