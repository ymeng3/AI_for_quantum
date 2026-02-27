#!/usr/bin/env python3
"""
Bradley-Terry Reward Model for RHEED Image Classification

This model learns to assign scalar "reward" scores to images for each reconstruction type.
Given pairwise preferences (image A is preferred over image B for type X), the model learns
scores r(A) and r(B) such that:

    P(A > B) = sigmoid(r(A) - r(B))

This is the same formulation used in RLHF for training reward models from human preferences.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models


# Reconstruction types (same as labeling software)
RECONSTRUCTION_TYPES = [
    '(1 x 1)',
    'Twinned(2 x 1)',
    'c(6 x 2)',
    '(√13 x √13)',
    'HTR'
]
NUM_TYPES = len(RECONSTRUCTION_TYPES)


class ImageEncoder(nn.Module):
    """
    Shared image encoder using ResNet18 backbone.
    Can optionally load pretrained weights from Classifier1's SimCLR encoder.
    """

    def __init__(self, pretrained_path=None):
        super().__init__()

        # Use ResNet18 backbone (same as Classifier1)
        backbone = models.resnet18(pretrained=False)
        # Remove final classification layer, keep up to global average pooling
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)  # Output: [B, 512, 1, 1]

        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained encoder from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] tensor of images

        Returns:
            [B, 512] embedding tensor
        """
        h = self.encoder(x)
        h = h.squeeze(-1).squeeze(-1)  # [B, 512]
        return h


class BradleyTerryRewardModel(nn.Module):
    """
    Reward model that outputs a scalar score for each reconstruction type.

    For each image, outputs 5 scores (one per reconstruction type).
    Higher score = more likely to be that reconstruction type.
    """

    def __init__(self, pretrained_encoder_path=None, hidden_dim=256):
        super().__init__()

        # Shared image encoder
        self.encoder = ImageEncoder(pretrained_encoder_path)

        # Reward head: embed -> hidden -> scores
        self.reward_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_TYPES)  # One score per reconstruction type
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] tensor of images

        Returns:
            [B, NUM_TYPES] tensor of reward scores
        """
        h = self.encoder(x)
        rewards = self.reward_head(h)
        return rewards

    def get_preference_probs(self, x1, x2, type_idx=None):
        """
        Compute P(x1 > x2) for specified reconstruction type(s).

        Args:
            x1: [B, 3, 224, 224] first images
            x2: [B, 3, 224, 224] second images
            type_idx: Optional[int] reconstruction type index.
                      If None, returns probs for all types.

        Returns:
            [B, 1] or [B, NUM_TYPES] tensor of probabilities
        """
        r1 = self.forward(x1)  # [B, NUM_TYPES]
        r2 = self.forward(x2)  # [B, NUM_TYPES]

        if type_idx is not None:
            r1 = r1[:, type_idx:type_idx+1]
            r2 = r2[:, type_idx:type_idx+1]

        # Bradley-Terry probability
        probs = torch.sigmoid(r1 - r2)
        return probs


def bradley_terry_loss(r_chosen, r_rejected):
    """
    Bradley-Terry loss for preference learning.

    Args:
        r_chosen: [B] tensor of rewards for preferred images
        r_rejected: [B] tensor of rewards for non-preferred images

    Returns:
        Scalar loss
    """
    # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    # We maximize this, so minimize -log(sigmoid(r_chosen - r_rejected))
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def tie_loss(r1, r2, margin=0.1):
    """
    Loss for tie cases: scores should be similar.

    Args:
        r1: [B] tensor of rewards for image 1
        r2: [B] tensor of rewards for image 2
        margin: Maximum allowed difference for ties

    Returns:
        Scalar loss
    """
    # Encourage |r1 - r2| < margin using hinge loss
    diff = torch.abs(r1 - r2)
    return F.relu(diff - margin).mean()


def not_apply_loss(r1, r2, low_threshold=-1.0):
    """
    Loss for 'not_apply' cases: both images should have low scores.

    Args:
        r1: [B] tensor of rewards for image 1
        r2: [B] tensor of rewards for image 2
        low_threshold: Target low score

    Returns:
        Scalar loss
    """
    # Encourage both scores to be below threshold
    loss1 = F.relu(r1 - low_threshold).mean()
    loss2 = F.relu(r2 - low_threshold).mean()
    return loss1 + loss2


class PairwiseDataset(torch.utils.data.Dataset):
    """
    Dataset for pairwise preference data.

    Each sample contains:
    - image1, image2: The two compared images
    - reconstruction_type: Index of the reconstruction type
    - winner: 0 = image1, 1 = image2, 2 = tie, 3 = not_apply
    """

    def __init__(self, df, image_root, transform=None):
        """
        Args:
            df: DataFrame with pairwise comparison data
            image_root: Root directory containing images
            transform: Optional image transform
        """
        self.df = df
        self.image_root = Path(image_root)

        # Default transform
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            ])
        else:
            self.transform = transform

        # Map reconstruction types to indices
        self.type_to_idx = {t: i for i, t in enumerate(RECONSTRUCTION_TYPES)}

        # Map winner strings to integers
        self.winner_map = {
            'image1': 0,
            'image2': 1,
            'tie': 2,
            'not_apply': 3
        }

        # Filter to valid entries only
        self._filter_valid_entries()

    def _filter_valid_entries(self):
        """Filter to entries with valid images and reconstruction types."""
        valid_indices = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]

            # Check reconstruction type is valid
            rt = row.get('reconstruction_type', row.get('Reconstruction_Type', ''))
            if rt not in self.type_to_idx:
                continue

            # Check winner is valid
            winner = row.get('winner', row.get('Winner', ''))
            if winner not in self.winner_map:
                continue

            valid_indices.append(idx)

        print(f"Valid entries: {len(valid_indices)} / {len(self.df)}")
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def _load_image(self, path):
        """Load and transform an image."""
        # Try to find the image
        if os.path.exists(path):
            img_path = path
        else:
            # Try relative to image_root
            img_path = self.image_root / path
            if not img_path.exists():
                # Try just the filename
                filename = Path(path).name
                img_path = self.image_root / filename

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(img_path).convert('L')  # Grayscale
        return self.transform(img)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]

        # Get image paths
        path1 = row.get('image1_path', row.get('Image1_Path', ''))
        path2 = row.get('image2_path', row.get('Image2_Path', ''))

        # Load images
        img1 = self._load_image(path1)
        img2 = self._load_image(path2)

        # Get reconstruction type index
        rt = row.get('reconstruction_type', row.get('Reconstruction_Type', ''))
        type_idx = self.type_to_idx[rt]

        # Get winner
        winner = row.get('winner', row.get('Winner', ''))
        winner_idx = self.winner_map[winner]

        return {
            'image1': img1,
            'image2': img2,
            'type_idx': type_idx,
            'winner': winner_idx,
            'path1': path1,
            'path2': path2
        }


def train_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        img1 = batch['image1'].to(device)
        img2 = batch['image2'].to(device)
        type_idx = batch['type_idx'].to(device)
        winner = batch['winner'].to(device)

        # Get rewards for both images
        r1 = model(img1)  # [B, NUM_TYPES]
        r2 = model(img2)  # [B, NUM_TYPES]

        # Gather rewards for the specific reconstruction types
        batch_size = img1.shape[0]
        r1_type = r1[torch.arange(batch_size, device=device), type_idx]
        r2_type = r2[torch.arange(batch_size, device=device), type_idx]

        # Compute loss based on winner type
        loss = torch.tensor(0.0, device=device)

        # image1 wins: r1 > r2
        mask_img1 = (winner == 0)
        if mask_img1.any():
            loss = loss + bradley_terry_loss(
                r1_type[mask_img1],
                r2_type[mask_img1]
            ) * mask_img1.sum() / batch_size

        # image2 wins: r2 > r1
        mask_img2 = (winner == 1)
        if mask_img2.any():
            loss = loss + bradley_terry_loss(
                r2_type[mask_img2],
                r1_type[mask_img2]
            ) * mask_img2.sum() / batch_size

        # tie: r1 ≈ r2
        mask_tie = (winner == 2)
        if mask_tie.any():
            loss = loss + tie_loss(
                r1_type[mask_tie],
                r2_type[mask_tie]
            ) * mask_tie.sum() / batch_size

        # not_apply: both low
        mask_na = (winner == 3)
        if mask_na.any():
            loss = loss + not_apply_loss(
                r1_type[mask_na],
                r2_type[mask_na]
            ) * mask_na.sum() / batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, dataloader, device):
    """
    Evaluate model accuracy on preference prediction.

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            type_idx = batch['type_idx'].to(device)
            winner = batch['winner'].to(device)

            # Get rewards
            r1 = model(img1)
            r2 = model(img2)

            batch_size = img1.shape[0]
            r1_type = r1[torch.arange(batch_size, device=device), type_idx]
            r2_type = r2[torch.arange(batch_size, device=device), type_idx]

            # Predict winner: image1 if r1 > r2, image2 otherwise
            pred = (r1_type <= r2_type).long()  # 0 = img1, 1 = img2

            # Only count non-tie, non-not_apply cases
            valid_mask = (winner == 0) | (winner == 1)
            correct += ((pred == winner) & valid_mask).sum().item()
            total += valid_mask.sum().item()

    accuracy = correct / total if total > 0 else 0
    return {'accuracy': accuracy, 'correct': correct, 'total': total}


if __name__ == '__main__':
    # Quick test
    print("Bradley-Terry Reward Model for RHEED Classification")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = BradleyTerryRewardModel()
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(4, 3, 224, 224).to(device)
    rewards = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {rewards.shape}")
    print(f"Sample rewards: {rewards[0].detach().cpu().numpy()}")

    # Test preference prediction
    x1 = torch.randn(4, 3, 224, 224).to(device)
    x2 = torch.randn(4, 3, 224, 224).to(device)
    probs = model.get_preference_probs(x1, x2, type_idx=0)
    print(f"Preference probs (type 0): {probs.squeeze().detach().cpu().numpy()}")

    print("\nModel architecture:")
    print(model)
