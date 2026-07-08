"""
train_emcdr.py
==============
Trains the EMCDR cross-domain mapping function.

Architecture: a 2-layer MLP that maps a user's Movies & TV BPR-MF embedding
into the Books BPR-MF embedding space, trained on overlapping users.

At inference time, ANY user with a Movies embedding (including cold-start users
not in the overlapping set) can have their embedding projected into Books space
for ranking.

Outputs saved to <DATA_DIR>/processed/emcdr/:
  mapping_mlp.pt          - trained MLP state dict (PyTorch)
  mapping_mlp_config.pkl  - model config (input_dim, hidden_dim, output_dim)

Usage in Jupyter:
  Edit the Configuration block, then run.

Requirements:
  pip install torch numpy
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = r'D:\Study\HEC Paris\Courses\Research\Research Thesis\Data'
HIDDEN_DIM = 128     # MLP hidden layer size
DROPOUT    = 0.2
EPOCHS     = 100
LR         = 1e-3
BATCH_SIZE = 256
SEED       = 42
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

PROC_DIR = os.path.join(DATA_DIR, 'processed')
BPR_DIR  = os.path.join(PROC_DIR, 'bprmf')
OUT_DIR  = os.path.join(PROC_DIR, 'emcdr')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# MLP mapping model
# ---------------------------------------------------------------------------

class MappingMLP(nn.Module):
    """
    Two-layer MLP: source_emb -> hidden -> hidden -> target_emb
    Maps Movies user embedding to Books user embedding space.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingPairDataset(Dataset):
    def __init__(self, source_embs, target_embs):
        self.source = torch.tensor(source_embs, dtype=torch.float32)
        self.target = torch.tensor(target_embs, dtype=torch.float32)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_emcdr():
    print("=" * 60)
    print("EMCDR Mapping MLP Training")
    print("=" * 60)

    # Load BPR-MF embeddings and index mappings
    print("\n[1] Loading BPR-MF embeddings ...")
    movies_user_factors = np.load(os.path.join(BPR_DIR, 'movies_user_factors.npy'))
    books_user_factors  = np.load(os.path.join(BPR_DIR, 'books_user_factors.npy'))
    movies_user2idx     = load_pkl(os.path.join(BPR_DIR, 'movies_user2idx.pkl'))
    books_user2idx      = load_pkl(os.path.join(BPR_DIR, 'books_user2idx.pkl'))

    print(f"  Movies user factors : {movies_user_factors.shape}")
    print(f"  Books user factors  : {books_user_factors.shape}")

    # Load overlapping users
    print("\n[2] Loading overlapping users ...")
    overlap_users = load_pkl(os.path.join(PROC_DIR, 'overlap_users.pkl'))
    print(f"  Total overlap users : {len(overlap_users):,}")

    # Filter to users present in both BPR-MF index maps
    valid_overlap = [
        uid for uid in overlap_users
        if uid in movies_user2idx and uid in books_user2idx
    ]
    print(f"  Overlap users with BPR-MF embeddings : {len(valid_overlap):,}")

    if len(valid_overlap) < 10:
        print("  WARNING: very few overlapping users. EMCDR mapping may be unreliable.")

    # Build paired embedding arrays
    src_embs = np.array([movies_user_factors[movies_user2idx[u]] for u in valid_overlap])
    tgt_embs = np.array([books_user_factors[books_user2idx[u]]   for u in valid_overlap])
    print(f"  Training pairs shape : {src_embs.shape}  ->  {tgt_embs.shape}")

    # Train/val split (90/10)
    n = len(valid_overlap)
    n_train = int(0.9 * n)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = EmbeddingPairDataset(src_embs[train_idx], tgt_embs[train_idx])
    val_ds   = EmbeddingPairDataset(src_embs[val_idx],   tgt_embs[val_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train / Val : {len(train_ds)} / {len(val_ds)}")

    # Model
    input_dim  = src_embs.shape[1]
    output_dim = tgt_embs.shape[1]
    model = MappingMLP(input_dim, HIDDEN_DIM, output_dim, DROPOUT)
    device = torch.device('cpu')   # CPU only
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\n[3] Training MLP (input={input_dim}, hidden={HIDDEN_DIM}, output={output_dim}) ...")
    print(f"  Device : {device}  |  Epochs : {EPOCHS}  |  LR : {LR}  |  Batch : {BATCH_SIZE}")

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(src)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(src)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(device), tgt.to(device)
                pred = model(src)
                val_loss += criterion(pred, tgt).item() * len(src)
        val_loss /= max(len(val_ds), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}"
                  + (" *" if val_loss == best_val_loss else ""))

    print(f"\n  Best val loss : {best_val_loss:.6f}")

    # Save best model
    config = {'input_dim': input_dim, 'hidden_dim': HIDDEN_DIM,
              'output_dim': output_dim, 'dropout': DROPOUT}
    torch.save(best_state, os.path.join(OUT_DIR, 'mapping_mlp.pt'))
    save_pkl(config, os.path.join(OUT_DIR, 'mapping_mlp_config.pkl'))

    print("\n" + "=" * 60)
    print("EMCDR TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to : {OUT_DIR}")
    print("Next step: run run_zero_shot.py and run_bridge.py for LLM conditions.")


train_emcdr()
