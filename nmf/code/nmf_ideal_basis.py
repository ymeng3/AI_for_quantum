"""
NMF with Ideal Image Basis

Train NMF on ideal reconstruction images (1x1, RT13, HTR, c6x2) to get
physically meaningful, fixed basis components. Then project trajectory
data onto this basis for consistent decomposition across runs.

This solves the component alignment problem by defining components from
known physical reconstructions rather than letting NMF find arbitrary bases.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageEnhance, ImageDraw, ImageStat
from sklearn.decomposition import NMF
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
IDEAL_DIR = Path(__file__).parent / "ideal_images"
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"
OUTPUT_DIR = Path(__file__).parent / "nmf_ideal_basis_results"

# Preprocessing params (same as original NMF notebook)
CONTRAST_FACTOR = 10
MASK_RECT = [280, 200, 310, 250]  # Specular reflection mask
TARGET_SIZE = (640, 480)  # Resize all images to same size

# Reconstruction classes
CLASSES = ['STO_ideal_1x1', 'STO_ideal_RT13', 'STO_ideal_HTR', 'STO_ideal_c6x2']
CLASS_LABELS = ['1x1', 'RT13', 'HTR', 'c6x2']

# Trajectories to analyze (path, display_name, type)
TRAJECTORIES = {
    'HTR': [
        ('2025-10-04/A', 'HL251004A'),
        ('2025-10-04/B', 'HL251004B'),
        ('2022-02-06', 'RR220206A'),
    ],
    'RT13': [
        ('2025-10-05', 'HL251005A'),
        ('2022-02-04', 'RR220204A'),
        ('2022-04-11', 'RR220411A'),
    ]
}


def load_and_preprocess_image(img_path, target_size=TARGET_SIZE):
    """Load and preprocess a single image."""
    img = Image.open(img_path).convert('RGB')

    # Resize to target size
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Mask specular reflection
    if MASK_RECT:
        draw = ImageDraw.Draw(img)
        draw.rectangle(MASK_RECT, fill=(0, 0, 0))

    # Contrast enhancement
    def change_contrast(img, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            return max(0, min(255, int(128 + factor * (c - 128))))
        return img.point(contrast)

    img = change_contrast(img, CONTRAST_FACTOR)

    return img


def load_ideal_images():
    """Load all ideal images from each reconstruction class."""
    images = []
    labels = []

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = IDEAL_DIR / cls_name
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} not found")
            continue

        files = list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.bmp'))
        print(f"Loading {len(files)} images from {cls_name}")

        for f in files:
            try:
                img = load_and_preprocess_image(f)
                images.append(img)
                labels.append(cls_idx)
            except Exception as e:
                print(f"  Error loading {f}: {e}")

    return images, labels


def load_trajectory(traj_path):
    """Load all images from a trajectory."""
    traj_dir = TRAJECTORY_DIR / traj_path

    if not traj_dir.exists():
        print(f"Warning: Trajectory {traj_path} not found at {traj_dir}")
        return [], []

    files = sorted(list(traj_dir.glob('*.bmp')) + list(traj_dir.glob('*.png')))
    images = []

    for f in files:
        try:
            img = load_and_preprocess_image(f)
            images.append(img)
        except Exception as e:
            pass  # Skip problematic images silently

    print(f"Loaded {len(images)} images from {traj_path}")
    return images, files


def images_to_matrix(images):
    """Convert list of PIL images to data matrix."""
    arrays = []
    for img in images:
        arr = np.array(img, dtype=np.float64) / 255.0
        arrays.append(arr.flatten())
    return np.array(arrays)


def train_nmf_on_ideals(images, n_components=4, max_iter=500):
    """Train NMF on ideal images to get fixed basis."""
    V = images_to_matrix(images)
    print(f"Training NMF on {V.shape[0]} ideal images, {n_components} components...")

    nmf = NMF(
        n_components=n_components,
        init='nndsvda',
        max_iter=max_iter,
        random_state=42
    )

    W = nmf.fit_transform(V)
    H = nmf.components_

    print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")
    return nmf, W, H


def project_trajectory(nmf, images):
    """Project trajectory images onto fixed NMF basis."""
    V = images_to_matrix(images)
    W = nmf.transform(V)
    return W


def plot_basis_components(H, img_size=TARGET_SIZE, output_path=None):
    """Visualize the NMF basis components."""
    n_components = H.shape[0]
    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 5))

    for i, ax in enumerate(axes):
        component = H[i].reshape(img_size[1], img_size[0], 3)
        component = component / component.max() if component.max() > 0 else component
        ax.imshow(component)
        ax.set_title(f'Component {i+1}', fontsize=14)
        ax.axis('off')

    plt.suptitle('NMF Basis Components (trained on ideal images)', fontsize=16)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def plot_ideal_decomposition(W, labels, output_path=None):
    """Show how ideal images decompose - verify each class maps to a component."""
    n_components = W.shape[1]

    # Compute mean weights per class
    mean_weights = np.zeros((len(CLASSES), n_components))
    for cls_idx in range(len(CLASSES)):
        mask = np.array(labels) == cls_idx
        if mask.sum() > 0:
            mean_weights[cls_idx] = W[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CLASSES))
    width = 0.2

    for i in range(n_components):
        ax.bar(x + i * width, mean_weights[:, i], width, label=f'Component {i+1}')

    ax.set_xlabel('Reconstruction Class', fontsize=12)
    ax.set_ylabel('Mean NMF Weight', fontsize=12)
    ax.set_xticks(x + width * (n_components - 1) / 2)
    ax.set_xticklabels(CLASS_LABELS)
    ax.legend()
    ax.set_title('Mean Component Weights by Reconstruction Class', fontsize=14)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()

    # Print component-class mapping
    print("\nComponent-Class Mapping (dominant component per class):")
    for cls_idx, cls_name in enumerate(CLASS_LABELS):
        dominant = np.argmax(mean_weights[cls_idx])
        print(f"  {cls_name}: Component {dominant + 1} (weight={mean_weights[cls_idx, dominant]:.3f})")

    return mean_weights


def plot_trajectory_comparison(trajectories_data, output_path=None):
    """Compare component evolution for HTR vs RT13 trajectories."""
    n_components = list(trajectories_data.values())[0]['W'].shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for row, (traj_type, traj_list) in enumerate(TRAJECTORIES.items()):
        for col, (traj_path, traj_name) in enumerate(traj_list):
            ax = axes[row, col]
            key = f"{traj_path.replace('/', '_')}_{traj_name}"

            if key not in trajectories_data:
                ax.text(0.5, 0.5, f'No data for {traj_name}', ha='center', va='center',
                       transform=ax.transAxes)
                continue

            W = trajectories_data[key]['W']

            # Normalize weights to sum to 1 for each frame
            W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)

            for i in range(n_components):
                ax.plot(W_norm[:, i], color=colors[i],
                       label=f'C{i+1}', linewidth=1.5, alpha=0.8)

            ax.set_title(f'{traj_name} ({traj_type})', fontsize=12)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Normalized Weight')
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Trajectory Decomposition with Ideal-Image NMF Basis\n'
                 '(Top: HTR trajectories, Bottom: RT13 trajectories)', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def plot_stacked_comparison(trajectories_data, component_labels, output_path=None):
    """Stacked area plot comparing HTR vs RT13 trajectories."""
    n_components = list(trajectories_data.values())[0]['W'].shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for row, (traj_type, traj_list) in enumerate(TRAJECTORIES.items()):
        for col, (traj_path, traj_name) in enumerate(traj_list):
            ax = axes[row, col]
            key = f"{traj_path.replace('/', '_')}_{traj_name}"

            if key not in trajectories_data:
                continue

            W = trajectories_data[key]['W']
            W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)

            ax.stackplot(range(len(W_norm)),
                        [W_norm[:, i] for i in range(n_components)],
                        labels=[f'C{i+1}: {component_labels.get(i, "?")}'
                               for i in range(n_components)],
                        colors=colors[:n_components],
                        alpha=0.7)

            ax.set_title(f'{traj_name} ({traj_type})', fontsize=12)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Composition')
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Reconstruction Composition (Stacked)\n'
                 '(Top: HTR trajectories, Bottom: RT13 trajectories)', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Load ideal images
    print("=" * 60)
    print("Step 1: Loading ideal images")
    print("=" * 60)
    ideal_images, ideal_labels = load_ideal_images()
    print(f"Total ideal images: {len(ideal_images)}")

    # Step 2: Train NMF on ideal images
    print("\n" + "=" * 60)
    print("Step 2: Training NMF on ideal images")
    print("=" * 60)
    nmf, W_ideal, H = train_nmf_on_ideals(ideal_images, n_components=4)

    # Step 3: Visualize basis components
    print("\n" + "=" * 60)
    print("Step 3: Visualizing basis components")
    print("=" * 60)
    plot_basis_components(H, output_path=OUTPUT_DIR / "01_basis_components.png")

    # Step 4: Verify ideal image decomposition
    print("\n" + "=" * 60)
    print("Step 4: Verifying ideal image decomposition")
    print("=" * 60)
    mean_weights = plot_ideal_decomposition(W_ideal, ideal_labels,
                                            output_path=OUTPUT_DIR / "02_ideal_decomposition.png")

    # Step 5: Project trajectories
    print("\n" + "=" * 60)
    print("Step 5: Projecting trajectories onto ideal basis")
    print("=" * 60)

    trajectories_data = {}

    for traj_type, traj_list in TRAJECTORIES.items():
        print(f"\n--- {traj_type} Trajectories ---")
        for traj_path, traj_name in traj_list:
            images, files = load_trajectory(traj_path)
            if len(images) > 0:
                W = project_trajectory(nmf, images)
                key = f"{traj_path.replace('/', '_')}_{traj_name}"
                trajectories_data[key] = {
                    'W': W,
                    'n_frames': len(images),
                    'type': traj_type,
                    'name': traj_name
                }
                print(f"  {traj_name}: {len(images)} frames projected")

    # Step 6: Compare trajectories
    if trajectories_data:
        print("\n" + "=" * 60)
        print("Step 6: Comparing HTR vs RT13 trajectories")
        print("=" * 60)

        # Create component labels based on which class each component best represents
        component_labels = {}
        for comp_idx in range(mean_weights.shape[1]):
            best_class_idx = np.argmax(mean_weights[:, comp_idx])
            component_labels[comp_idx] = CLASS_LABELS[best_class_idx]

        print(f"Component labels: {component_labels}")

        plot_trajectory_comparison(trajectories_data,
                                   output_path=OUTPUT_DIR / "03_trajectory_comparison.png")
        plot_stacked_comparison(trajectories_data, component_labels,
                               output_path=OUTPUT_DIR / "04_stacked_comparison.png")

    print("\n" + "=" * 60)
    print("Done! Results saved to:", OUTPUT_DIR)
    print("=" * 60)

    return nmf, trajectories_data, mean_weights


if __name__ == "__main__":
    nmf, trajectories_data, mean_weights = main()
