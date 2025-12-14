#!/usr/bin/env python3
"""
Simple inference script to classify RHEED images using the trained encoder.
Usage: python classify_image.py <image_path>
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# Import functions from classification.py
sys.path.insert(0, os.path.dirname(__file__))
from classification import SimCLRModel, to_canon_luminance, apply_center_mask, load_gray_any

def load_encoder(encoder_path, device='cpu'):
    """Load the trained encoder."""
    model = SimCLRModel(proj_dim=128).to(device)
    state = torch.load(encoder_path, map_location=device)
    model.encoder.load_state_dict(state)
    model.eval()
    return model

def encode_image(image_path, encoder, device='cpu', mask_center=True):
    """Encode a single image using the trained encoder."""
    img = to_canon_luminance(image_path, mask_center=mask_center)
    x = T.ToTensor()(img).unsqueeze(0)  # [1,1,224,224]
    x = x.repeat(1, 3, 1, 1)  # [1,3,224,224]
    x = T.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])(x)
    x = x.to(device)
    
    with torch.no_grad():
        h = encoder(x).squeeze().cpu().numpy()  # [512]
        h = h / (np.linalg.norm(h) + 1e-12)  # L2 normalize
    return h

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_image.py <image_path> [encoder_path]")
        print("Example: python classify_image.py data/Test/image.bmp")
        sys.exit(1)
    
    image_path = sys.argv[1]
    encoder_path = sys.argv[2] if len(sys.argv) > 2 else "src/artifacts/encoders/simclr_resnet18_encoder.pth"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder not found: {encoder_path}")
        print("Available encoders:")
        encoders_dir = "src/artifacts/encoders"
        if os.path.exists(encoders_dir):
            for f in os.listdir(encoders_dir):
                if f.endswith('.pth'):
                    print(f"  - {os.path.join(encoders_dir, f)}")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading encoder from: {encoder_path}")
    
    encoder = load_encoder(encoder_path, device)
    print(f"Encoding image: {image_path}")
    
    embedding = encode_image(image_path, encoder.encoder, device)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"First 10 values: {embedding[:10]}")
    print("\n✓ Image encoded successfully!")
    print("Note: This is just the embedding. For full classification, you need the k-NN classifier.")

if __name__ == "__main__":
    main()

