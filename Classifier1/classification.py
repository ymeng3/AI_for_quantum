# ---------------- Encoder, dataset, training loop ----------------

# --- simclr_lite.py ---
import os, random, math, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# ---------------- Shared preprocessing ----------------
def apply_center_mask(gray: np.ndarray, frac: float = 0.60) -> np.ndarray:
    h, w = gray.shape
    cy, cx = h/2.0, w/2.0
    R = min(h,w) * frac * 0.5
    yy, xx = np.ogrid[:h,:w]
    mask = ((yy-cy)**2 + (xx-cx)**2) > R**2
    out = gray.copy()
    out[~mask] = np.median(gray)
    return out

def load_gray_any(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim > 2: arr = np.squeeze(arr)
        lo, hi = np.percentile(arr, [1,99]); hi = max(hi, lo+1e-3)
        arr = np.clip((arr - lo)/(hi - lo), 0, 1) * 255.0
        return arr
    else:
        return np.array(Image.open(path).convert("L"), dtype=np.float32)

# image -> (224,224) masked luminance (uint8)
def to_canon_luminance(path: str, mask_center=True) -> Image.Image:
    g = load_gray_any(path)
    if mask_center: g = apply_center_mask(g)
    g = np.array(Image.fromarray(g).resize((224,224))).astype(np.uint8)
    return Image.fromarray(g)

# ---------------- Dataset that returns two views ----------------
class RHEEDSimCLR(torch.utils.data.Dataset):
    def __init__(self, files, mask_center=True):
        self.files = files
        self.mask_center = mask_center
        # augmentations (keep small + physics-friendly)
        self.base = T.Compose([
            T.ToTensor(),  # -> [0,1]
            T.Normalize(mean=[0.5], std=[0.25]),
        ])
        self.aug = T.Compose([
            T.RandomAffine(degrees=7, translate=(0.05,0.05), scale=(0.95,1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.GaussianBlur(kernel_size=3, sigma=(0.1,1.0)),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = to_canon_luminance(p, mask_center=self.mask_center)  # PIL (L)
        # two correlated views: (aug -> to_tensor -> to3ch -> norm)
        def view(im):
            v = self.aug(im)  # PIL Image -> PIL Image (augmented)
            v = T.ToTensor()(v)  # PIL Image -> Tensor [0,1] [1,H,W]
            v = v.clamp(0,1)
            v3 = v.repeat(3,1,1)           # gray->3ch [3,H,W]
            v3 = T.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25])(v3)
            return v3
        x1, x2 = view(img), view(img)
        return x1, x2

# ---------------- Encoder + projection head ----------------
class SimCLRModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        # use torchvision's resnet18 but remove final fc
        modules = list(backbone.children())[:-1]  # up to global pool
        self.encoder = nn.Sequential(*modules)    # -> [B,512,1,1]
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )
    def forward(self, x):
        h = self.encoder(x).squeeze(-1).squeeze(-1)  # [B,512]
        z = self.proj(h)                              # [B,d]
        z = nn.functional.normalize(z, dim=1)
        return h, z

# ---------------- NT-Xent loss ----------------
def nt_xent_loss(z1, z2, T=0.2):
    z = torch.cat([z1, z2], dim=0)                 # [2B,d]
    z = nn.functional.normalize(z, dim=1)
    N = z.size(0)
    B = N // 2
    sim = torch.mm(z, z.t()) / T                   # cosine / T
    mask = torch.eye(N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)                  # remove self-similarity
    # positives: sample i pairs with sample i+B (same image, different aug)
    # For samples 0..B-1, positive is B..2B-1
    # For samples B..2B-1, positive is 0..B-1
    labels = torch.cat([torch.arange(B, 2*B, device=z.device),
                        torch.arange(0, B, device=z.device)])
    loss = nn.CrossEntropyLoss()(sim, labels)
    return loss



# ---------------- Training loop ----------------

# --- train_simclr.py ---
import torch, torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path

# Build a file list from your 5 trajectories (subsample if huge)
def collect_files(root, subsample=3):
    import glob, os
    files = []
    for ext in ("*.png","*.bmp","*.tif","*.tiff","*.npy","*.jpg","*.jpeg"):
        files += glob.glob(os.path.join(root, "**", ext), recursive=True)
    files = sorted(files)
    if subsample and subsample>1:
        files = files[::subsample]
    return files

def train_model():
    """Main training function - only runs when script is executed directly."""
    # Point to the Trajectories directory (contains 5 trajectory folders) 
    root = "/home/ymeng3/project_quantum/data/Trajectories" 
    files = collect_files(root, subsample=3)  # e.g., ~5–10k frames 
    print(f"train frames: {len(files)}") 
    print(f"Sample files: {files[:3]}") 

    ds = RHEEDSimCLR(files, mask_center=True) 
    # num_workers=0 for Jupyter notebooks to avoid multiprocessing pickling issues 
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=False) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Using device: {device}") 
    model = SimCLRModel(proj_dim=128).to(device) 
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) 

    # Create artifacts directory if it doesn't exist 
    Path("src/artifacts/encoders").mkdir(parents=True, exist_ok=True) 

    epochs = 40 
    num_batches = len(dl) 
    print(f"\nStarting training: {epochs} epochs, {num_batches} batches per epoch") 
    print("=" * 60) 

    start_time = time.time() 
    for ep in range(1, epochs+1): 
        model.train() 
        running = 0.0 
        epoch_start = time.time() 
        
        for batch_idx, (x1, x2) in enumerate(dl, 1): 
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True) 
            h1, z1 = model(x1) 
            h2, z2 = model(x2) 
            loss = nt_xent_loss(z1, z2, T=0.2) 
            opt.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += float(loss)
            
            # Log progress every 10% of batches or every 10 batches, whichever is more frequent
            if batch_idx % max(1, num_batches // 10) == 0 or batch_idx % 10 == 0:
                progress = 100 * batch_idx / num_batches
                avg_loss = running / batch_idx
                print(f"Epoch {ep:03d}/{epochs} | Batch {batch_idx:4d}/{num_batches} ({progress:5.1f}%) | Loss: {avg_loss:.4f}", end='\r', flush=True)
        
        epoch_time = time.time() - epoch_start
        avg_loss = running / num_batches
        elapsed_time = time.time() - start_time
        eta = (elapsed_time / ep) * (epochs - ep) if ep > 0 else 0
        
        print(f"Epoch {ep:03d}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}m" + " " * 20)

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes ({total_time:.1f} seconds)")

    # save encoder (without projection head) for downstream
    save_path = "src/artifacts/encoders/simclr_resnet18_encoder.pth"
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Encoder saved to: {save_path}")

if __name__ == "__main__":
    train_model()









# ---------------- Embed and retrieve ----------------


# # --- embed_and_retrieve.py ---
# import numpy as np, torch
# from PIL import Image
# import torchvision.transforms as T
# import faiss

# # load encoder
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimCLRModel(proj_dim=128).to(device)
# # load weights but only keep encoder; set to eval
# state = torch.load("artifacts/encoders/simclr_resnet18_encoder.pth", map_location=device)
# model.encoder.load_state_dict(state)
# model.proj = torch.nn.Identity()
# model.eval()

# def encode_paths(paths):
#     vecs = []
#     with torch.no_grad():
#         for p in paths:
#             img = to_canon_luminance(p, mask_center=True)
#             x = T.ToTensor()(img).unsqueeze(0)            # [1,1,224,224]
#             x = x.repeat(1,3,1,1)
#             x = T.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25])(x)
#             x = x.to(device)
#             h = model.encoder(x).squeeze().cpu().numpy()  # [512]
#             h = h / (np.linalg.norm(h)+1e-12)
#             vecs.append(h.astype('float32'))
#     return np.vstack(vecs)

# # build index for retrieval
# all_paths = [...]  # point to all frames or a big subset
# X = encode_paths(all_paths)
# index = faiss.IndexFlatIP(X.shape[1])  # cosine via inner product (since L2-normed)
# index.add(X)

# # query with a seed (e.g., a known RT13 frame)
# seed = "/path/to/a/rt13_seed.png"
# q = encode_paths([seed])
# D, I = index.search(q, 10)
# print("Top-10 neighbors:")
# for rank, idx in enumerate(I[0]):
#     print(rank+1, all_paths[idx], D[0][rank])
