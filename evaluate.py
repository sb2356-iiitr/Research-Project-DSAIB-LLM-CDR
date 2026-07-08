"""
evaluate.py
===========
Computes NDCG@10, HR@10, and MRR for all models across all cold-start splits.

Reads ranking outputs from:
  processed/results/zero_shot/{split}_rankings.pkl
  processed/results/bridge/{split}_rankings.pkl

Also computes BPR-MF and EMCDR baselines directly (no separate run script
needed -- they rank by embedding dot product against candidate items).

Outputs:
  processed/results/evaluation_results.pkl  - full results dict
  processed/results/evaluation_results.csv  - human-readable table

Usage: edit Configuration block and run.
"""

import os
import pickle
import random
import numpy as np
import csv
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR     = r'D:\Study\HEC Paris\Courses\Research\Research Thesis\Data'
SEED         = 42
# ---------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

PROC_DIR    = os.path.join(DATA_DIR, 'processed')
BPR_DIR     = os.path.join(PROC_DIR, 'bprmf')
EMCDR_DIR   = os.path.join(PROC_DIR, 'emcdr')
ZS_DIR      = os.path.join(PROC_DIR, 'results', 'zero_shot')
BRIDGE_DIR  = os.path.join(PROC_DIR, 'results', 'bridge')
OUT_DIR     = os.path.join(PROC_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked_list, positive, k=10):
    """NDCG@k for a single user. ranked_list is ordered best-to-worst."""
    try:
        rank = ranked_list.index(positive) + 1  # 1-indexed
    except ValueError:
        return 0.0
    if rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def hr_at_k(ranked_list, positive, k=10):
    """Hit Rate@k."""
    try:
        rank = ranked_list.index(positive) + 1
    except ValueError:
        return 0.0
    return 1.0 if rank <= k else 0.0


def mrr(ranked_list, positive):
    """Mean Reciprocal Rank (no cutoff)."""
    try:
        rank = ranked_list.index(positive) + 1
    except ValueError:
        return 0.0
    return 1.0 / rank


def compute_metrics(rankings, candidates):
    """
    rankings : {user_id: [ranked asin list]}
    candidates: {user_id: {positive: asin, negatives: [...]}}
    Returns dict with mean NDCG@10, HR@10, MRR.
    """
    ndcg_vals, hr_vals, mrr_vals = [], [], []
    for uid, ranked in rankings.items():
        if uid not in candidates:
            continue
        positive = candidates[uid]['positive']
        ndcg_vals.append(ndcg_at_k(ranked, positive, k=10))
        hr_vals.append(hr_at_k(ranked, positive, k=10))
        mrr_vals.append(mrr(ranked, positive))

    n = len(ndcg_vals)
    return {
        'NDCG@10': np.mean(ndcg_vals) if n > 0 else 0.0,
        'HR@10':   np.mean(hr_vals)   if n > 0 else 0.0,
        'MRR':     np.mean(mrr_vals)  if n > 0 else 0.0,
        'n_users': n,
    }


# ---------------------------------------------------------------------------
# BPR-MF baseline ranking (no cross-domain transfer)
# ---------------------------------------------------------------------------

def bprmf_rank(user_entry, positive, negatives,
               books_user2idx, books_user_factors, books_item2idx, books_item_factors,
               rng=None):
    """
    Rank candidates using Books BPR-MF dot product.
    Falls back to random order if user has no Books embedding (cold-start users
    are not in books_5core, so BPR-MF has no useful signal for them).
    """
    uid = user_entry['user_id']
    all_cands = [positive] + negatives

    if uid not in books_user2idx:
        # No target-domain embedding: random fallback (BPR-MF has no signal)
        shuffled = list(all_cands)
        if rng:
            rng.shuffle(shuffled)
        else:
            random.shuffle(shuffled)
        return shuffled

    u_idx = books_user2idx[uid]
    if u_idx >= len(books_user_factors):
        shuffled = list(all_cands)
        if rng:
            rng.shuffle(shuffled)
        else:
            random.shuffle(shuffled)
        return shuffled

    u_emb = books_user_factors[u_idx]
    scores = {}
    for asin in all_cands:
        if asin in books_item2idx:
            i_idx = books_item2idx[asin]
            if i_idx < len(books_item_factors):
                scores[asin] = float(np.dot(u_emb, books_item_factors[i_idx]))
            else:
                scores[asin] = 0.0
        else:
            scores[asin] = 0.0

    return sorted(all_cands, key=lambda a: scores[a], reverse=True)


# ---------------------------------------------------------------------------
# EMCDR baseline ranking
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

class MappingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def load_emcdr_model():
    config_path = os.path.join(EMCDR_DIR, 'mapping_mlp_config.pkl')
    model_path  = os.path.join(EMCDR_DIR, 'mapping_mlp.pt')
    if not os.path.exists(model_path):
        return None, None
    config = load_pkl(config_path)
    model  = MappingMLP(config['input_dim'], config['hidden_dim'],
                        config['output_dim'], config['dropout'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, config


def emcdr_rank(user_entry, positive, negatives,
               movies_user2idx, movies_user_factors,
               books_item2idx, books_item_factors,
               emcdr_model):
    """Map Movies embedding to Books space, then rank by dot product."""
    uid = user_entry['user_id']
    all_cands = [positive] + negatives

    if emcdr_model is None or uid not in movies_user2idx:
        return list(all_cands)

    m_idx = movies_user2idx[uid]
    if m_idx >= len(movies_user_factors):
        return list(all_cands)

    m_emb = torch.tensor(movies_user_factors[m_idx], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mapped_emb = emcdr_model(m_emb).squeeze(0).numpy()

    scores = {}
    for asin in all_cands:
        if asin in books_item2idx:
            i_idx = books_item2idx[asin]
            if i_idx < len(books_item_factors):
                scores[asin] = float(np.dot(mapped_emb, books_item_factors[i_idx]))
            else:
                scores[asin] = 0.0
        else:
            scores[asin] = 0.0

    return sorted(all_cands, key=lambda a: scores[a], reverse=True)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_baselines(split_name, users, candidates_ref,
                       books_user2idx, books_user_factors,
                       books_item2idx, books_item_factors,
                       movies_user2idx, movies_user_factors,
                       emcdr_model):
    """Compute BPR-MF and EMCDR rankings for users in candidates_ref."""
    bprmf_rankings = {}
    emcdr_rankings = {}
    rng = random.Random(SEED)   # deterministic fallback shuffle

    for entry in users:
        uid = entry['user_id']
        if uid not in candidates_ref:
            continue
        positive  = candidates_ref[uid]['positive']
        negatives = candidates_ref[uid]['negatives']

        bprmf_rankings[uid] = bprmf_rank(
            entry, positive, negatives,
            books_user2idx, books_user_factors,
            books_item2idx, books_item_factors,
            rng=rng
        )
        emcdr_rankings[uid] = emcdr_rank(
            entry, positive, negatives,
            movies_user2idx, movies_user_factors,
            books_item2idx, books_item_factors,
            emcdr_model
        )

    return bprmf_rankings, emcdr_rankings


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    print("=" * 60)
    print("Evaluation: NDCG@10, HR@10, MRR")
    print("=" * 60)

    print("\nLoading BPR-MF and EMCDR artifacts ...")
    books_user2idx     = load_pkl(os.path.join(BPR_DIR, 'books_user2idx.pkl'))
    books_user_factors = np.load( os.path.join(BPR_DIR, 'books_user_factors.npy'))
    books_item2idx     = load_pkl(os.path.join(BPR_DIR, 'books_item2idx.pkl'))
    books_item_factors = np.load( os.path.join(BPR_DIR, 'books_item_factors.npy'))
    movies_user2idx    = load_pkl(os.path.join(BPR_DIR, 'movies_user2idx.pkl'))
    movies_user_factors= np.load( os.path.join(BPR_DIR, 'movies_user_factors.npy'))

    emcdr_model, _ = load_emcdr_model()
    if emcdr_model is None:
        print("  WARNING: EMCDR model not found. EMCDR results will be skipped.")

    cold_start_splits = load_pkl(os.path.join(PROC_DIR, 'cold_start_splits.pkl'))

    SPLITS = ['pure', 'very_cold', 'sparse']
    MODELS = ['Zero-Shot LLM', 'LLM Bridge', 'BPR-MF', 'EMCDR']
    results = defaultdict(dict)  # results[model][split] = metrics dict

    for split_name in SPLITS:
        print(f"\n--- Split: {split_name} ---")
        users = cold_start_splits[split_name]

        # Load LLM candidates (same for both LLM models)
        zs_cands_path = os.path.join(ZS_DIR, f'{split_name}_candidates.pkl')
        if not os.path.exists(zs_cands_path):
            print(f"  No zero-shot results for {split_name}, skipping.")
            continue
        candidates_ref = load_pkl(zs_cands_path)

        # Zero-shot rankings
        zs_path = os.path.join(ZS_DIR, f'{split_name}_rankings.pkl')
        if os.path.exists(zs_path):
            zs_rankings = load_pkl(zs_path)
            results['Zero-Shot LLM'][split_name] = compute_metrics(zs_rankings, candidates_ref)
            m = results['Zero-Shot LLM'][split_name]
            print(f"  Zero-Shot LLM   | NDCG@10={m['NDCG@10']:.4f} HR@10={m['HR@10']:.4f} MRR={m['MRR']:.4f} (n={m['n_users']})")

        # Bridge rankings
        bridge_path = os.path.join(BRIDGE_DIR, f'{split_name}_rankings.pkl')
        if os.path.exists(bridge_path):
            bridge_rankings = load_pkl(bridge_path)
            results['LLM Bridge'][split_name] = compute_metrics(bridge_rankings, candidates_ref)
            m = results['LLM Bridge'][split_name]
            print(f"  LLM Bridge      | NDCG@10={m['NDCG@10']:.4f} HR@10={m['HR@10']:.4f} MRR={m['MRR']:.4f} (n={m['n_users']})")

        # BPR-MF and EMCDR baselines (use same candidates as zero-shot for fair comparison)
        users_eval = [e for e in users if e['user_id'] in candidates_ref]
        bprmf_r, emcdr_r = evaluate_baselines(
            split_name, users_eval, candidates_ref,
            books_user2idx, books_user_factors,
            books_item2idx, books_item_factors,
            movies_user2idx, movies_user_factors,
            emcdr_model
        )

        results['BPR-MF'][split_name] = compute_metrics(bprmf_r, candidates_ref)
        m = results['BPR-MF'][split_name]
        print(f"  BPR-MF          | NDCG@10={m['NDCG@10']:.4f} HR@10={m['HR@10']:.4f} MRR={m['MRR']:.4f} (n={m['n_users']})")

        if emcdr_model is not None:
            results['EMCDR'][split_name] = compute_metrics(emcdr_r, candidates_ref)
            m = results['EMCDR'][split_name]
            print(f"  EMCDR           | NDCG@10={m['NDCG@10']:.4f} HR@10={m['HR@10']:.4f} MRR={m['MRR']:.4f} (n={m['n_users']})")

    # Save full results
    save_pkl(dict(results), os.path.join(OUT_DIR, 'evaluation_results.pkl'))

    # Save CSV
    csv_path = os.path.join(OUT_DIR, 'evaluation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Split', 'NDCG@10', 'HR@10', 'MRR', 'N_Users'])
        for model in MODELS:
            for split in SPLITS:
                if model in results and split in results[model]:
                    m = results[model][split]
                    writer.writerow([model, split,
                                     f"{m['NDCG@10']:.4f}",
                                     f"{m['HR@10']:.4f}",
                                     f"{m['MRR']:.4f}",
                                     m['n_users']])

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {csv_path}")

    # Print final summary table
    print("\nSummary Table (NDCG@10):")
    header = f"{'Model':<18}" + "".join(f"  {s:<12}" for s in SPLITS)
    print(header)
    print("-" * len(header))
    for model in MODELS:
        row = f"{model:<18}"
        for split in SPLITS:
            if model in results and split in results[model]:
                row += f"  {results[model][split]['NDCG@10']:.4f}      "
            else:
                row += f"  {'N/A':<12}"
        print(row)


main()
