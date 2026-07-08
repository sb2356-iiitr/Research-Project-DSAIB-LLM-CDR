"""
train_bprmf.py
==============
Trains BPR-MF on Movies & TV and Books 5-core data using the 'implicit' library
(CPU-optimized, no GPU required). Saves user/item embeddings and index mappings.

Outputs saved to <DATA_DIR>/processed/bprmf/:
  movies_user_factors.npy   - shape (n_users, EMB_DIM)
  movies_item_factors.npy   - shape (n_items, EMB_DIM)
  movies_user2idx.pkl       - {user_id: int index}
  movies_item2idx.pkl       - {parent_asin: int index}
  books_user_factors.npy    - same for Books
  books_item_factors.npy
  books_user2idx.pkl
  books_item2idx.pkl

Usage in Jupyter:
  Edit the Configuration block, then run.

Requirements:
  pip install implicit scipy numpy
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = r'D:\Study\HEC Paris\Courses\Research\Research Thesis\Data'
EMB_DIM  = 64       # embedding dimension
EPOCHS   = 100      # training epochs
LR       = 0.01     # learning rate
REG      = 0.01     # L2 regularisation
NEG_SAMPLES = 4     # negative samples per positive
SEED     = 42
# ---------------------------------------------------------------------------

PROC_DIR = os.path.join(DATA_DIR, 'processed')
OUT_DIR  = os.path.join(PROC_DIR, 'bprmf')
os.makedirs(OUT_DIR, exist_ok=True)


def load_pkl(name):
    path = os.path.join(PROC_DIR, name)
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved -> {path}")


def build_sparse_matrix(user2items, user2idx, item2idx):
    """
    Build a user-item CSR matrix of shape (n_users, n_items).
    Values are implicit feedback (1.0 per interaction).
    implicit expects item-user matrix, so we return both.
    """
    rows, cols = [], []
    for uid, interactions in user2items.items():
        u = user2idx[uid]
        for (asin, ts, rating) in interactions:
            if asin in item2idx:
                rows.append(u)
                cols.append(item2idx[asin])
    data = np.ones(len(rows), dtype=np.float32)
    n_users = len(user2idx)
    n_items = len(item2idx)
    user_item = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    item_user = user_item.T.tocsr()  # implicit expects item-user
    return user_item, item_user


def train_domain(domain_name, pkl_name):
    print(f"\n{'='*60}")
    print(f"Training BPR-MF on {domain_name}")
    print(f"{'='*60}")

    user2items = load_pkl(pkl_name)

    # Build index mappings
    all_users = sorted(user2items.keys())
    all_items = sorted({asin for v in user2items.values() for (asin, _, _) in v})
    user2idx  = {u: i for i, u in enumerate(all_users)}
    item2idx  = {a: i for i, a in enumerate(all_items)}
    print(f"  Users: {len(user2idx):,}  |  Items: {len(item2idx):,}")

    # Build sparse matrix
    user_item, item_user = build_sparse_matrix(user2items, user2idx, item2idx)
    n_interactions = user_item.nnz
    print(f"  Interactions: {n_interactions:,}")
    density = 100 * n_interactions / (len(user2idx) * len(item2idx))
    print(f"  Matrix density: {density:.4f}%")

    # Train BPR-MF
    print(f"\n  Training BPR-MF (dim={EMB_DIM}, epochs={EPOCHS}, lr={LR}, reg={REG}) ...")
    model = BayesianPersonalizedRanking(
        factors=EMB_DIM,
        iterations=EPOCHS,
        learning_rate=LR,
        regularization=REG,
        num_threads=0,   # use all CPU cores
        random_state=SEED,
        verify_negative_samples=True,
    )
    model.fit(user_item, show_progress=True)   # implicit v0.5+ expects user-item matrix

    # Extract embeddings
    user_factors = np.array(model.user_factors)  # (n_users, EMB_DIM)
    item_factors = np.array(model.item_factors)  # (n_items, EMB_DIM)
    print(f"  User factors shape: {user_factors.shape}")
    print(f"  Item factors shape: {item_factors.shape}")

    # Save
    prefix = domain_name.lower().replace(' ', '_').replace('&', 'and').replace('__', '_')
    # Normalise prefix
    if 'movie' in prefix:
        prefix = 'movies'
    else:
        prefix = 'books'

    np.save(os.path.join(OUT_DIR, f'{prefix}_user_factors.npy'), user_factors)
    np.save(os.path.join(OUT_DIR, f'{prefix}_item_factors.npy'), item_factors)
    save_pkl(user2idx, os.path.join(OUT_DIR, f'{prefix}_user2idx.pkl'))
    save_pkl(item2idx, os.path.join(OUT_DIR, f'{prefix}_item2idx.pkl'))

    print(f"\n  {domain_name} BPR-MF complete.")
    return user2idx, item2idx, user_factors, item_factors


def main():
    print("BPR-MF Training")
    print(f"Output directory: {OUT_DIR}")

    train_domain('Movies & TV', 'movies_5core.pkl')
    train_domain('Books', 'books_5core.pkl')

    print("\n" + "=" * 60)
    print("BPR-MF TRAINING COMPLETE")
    print("=" * 60)
    print(f"Embeddings saved to: {OUT_DIR}")
    print("Next step: run train_emcdr.py to learn the cross-domain mapping.")


main()
