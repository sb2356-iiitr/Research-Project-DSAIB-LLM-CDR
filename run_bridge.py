"""
run_bridge.py
=============
Condition B: LLM as Knowledge Transfer Bridge.

Two-stage pipeline:
  Stage 1 - Profile generation:
    LLM reads the user's Movies & TV history and writes a structured
    natural-language preference profile (~150-250 words).

  Stage 2 - Embedding-based ranking:
    The profile and all 20 Books candidate descriptions are encoded with
    text-embedding-3-small. Candidates are ranked by cosine similarity
    to the profile embedding.

    For SPARSE users (3-5 Books interactions), an optional collaborative
    blend mixes the LLM embedding with the user's BPR-MF Books embedding
    at a configurable weight (default 0.5 / 0.5).

All LLM responses and embeddings are cached to disk for reproducibility
and cost efficiency.

Outputs saved to <DATA_DIR>/processed/results/bridge/:
  {split}_rankings.pkl    - {user_id: [ranked asin list]}
  {split}_candidates.pkl  - {user_id: {positive, negatives}}

Usage: edit the Configuration block and run.

Requirements:
  pip install openai numpy tqdm
"""

import os
import pickle
import random
import time
import numpy as np
from tqdm import tqdm
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR        = r'D:\Study\HEC Paris\Courses\Research\Research Thesis\Data'
OPENAI_API_KEY  = os.environ.get('OPENAI_API_KEY', 'sk-...')  # replace or set env var
CHAT_MODEL      = 'gpt-4o-mini'           # for preference profile generation
EMBED_MODEL     = 'text-embedding-3-small' # 1536-dim, cheap
N_CANDIDATES    = 20                       # 1 positive + 19 negatives
MAX_HISTORY     = 20                       # max movies in prompt
COLLAB_WEIGHT   = 0.5    # weight for BPR-MF embedding blend (sparse users only)
                          # 0.0 = pure LLM, 1.0 = pure BPR-MF
N_EVAL_USERS    = 500    # per split; set to 0 for all
SEED            = 42
# ---------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

PROC_DIR = os.path.join(DATA_DIR, 'processed')
BPR_DIR  = os.path.join(PROC_DIR, 'bprmf')
OUT_DIR  = os.path.join(PROC_DIR, 'results', 'bridge')
os.makedirs(OUT_DIR, exist_ok=True)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Candidate sampling (identical to zero-shot for fair comparison)
# ---------------------------------------------------------------------------

def sample_candidates(user_entry, all_book_asins, rng):
    seen_books = {e[0] for e in (user_entry.get('books_train') or [])}
    if user_entry.get('books_val'):
        seen_books.add(user_entry['books_val'][0])

    if user_entry.get('books_test'):
        positive = user_entry['books_test'][0]
    else:
        pool = list(all_book_asins - seen_books)
        positive = rng.choice(pool)

    seen_books.add(positive)
    neg_pool = list(all_book_asins - seen_books)
    negatives = rng.sample(neg_pool, min(N_CANDIDATES - 1, len(neg_pool)))
    return positive, negatives


# ---------------------------------------------------------------------------
# Item text builders
# ---------------------------------------------------------------------------

def movie_text(asin, meta):
    m = meta.get(asin, {})
    title  = m.get('title') or asin
    cats   = m.get('categories', [])
    genre  = ', '.join(cats[:3]) if cats else ''
    feats  = m.get('features', [])
    extra  = '; '.join(str(f) for f in feats[:3]) if feats else ''
    parts  = [title]
    if genre:
        parts.append(f"[{genre}]")
    if extra:
        parts.append(f"({extra})")
    return ' '.join(parts)


def book_text_for_embedding(asin, meta):
    """Rich text for embedding -- includes description."""
    m = meta.get(asin, {})
    parts = []
    title = m.get('title') or asin
    parts.append(f"Title: {title}")
    author = m.get('author', '')
    if author:
        parts.append(f"Author: {author}")
    cats = m.get('categories', [])
    if len(cats) > 1:
        parts.append(f"Genre: {', '.join(cats[1:4])}")
    desc = m.get('description', '')
    if desc:
        parts.append(f"Description: {desc[:300]}")
    return ' | '.join(parts)


def book_text_short(asin, meta):
    """Short text for ranking prompts."""
    m = meta.get(asin, {})
    title  = m.get('title') or asin
    author = m.get('author', '')
    cats   = m.get('categories', [])
    genre  = ', '.join(cats[1:3]) if len(cats) > 1 else ''
    parts  = [title]
    if author:
        parts.append(f"by {author}")
    if genre:
        parts.append(f"[{genre}]")
    return ' | '.join(parts)


# ---------------------------------------------------------------------------
# Stage 1: LLM preference profile generation
# ---------------------------------------------------------------------------

PROFILE_PROMPT_TEMPLATE = """\
A user has watched the following movies and TV shows (most recent first):

{history}

Based on this watch history, write a detailed user preference profile (150-250 words) that describes:
- Preferred genres and themes
- Narrative style preferences (e.g. plot-driven vs character-driven, pacing, tone)
- Emotional or intellectual engagement patterns
- Any inferred preferences that might transfer to book reading

Write the profile as a cohesive paragraph, not a list. Be specific and nuanced."""


def generate_profile(client, user_entry, meta_movies, cache):
    uid = user_entry['user_id']
    if uid in cache.get('profiles', {}):
        return cache['profiles'][uid]

    history = list(reversed(user_entry['movies_interactions']))[:MAX_HISTORY]
    history_text = '\n'.join(
        f"  {i+1}. {movie_text(asin, meta_movies)}"
        for i, (asin, ts, rating) in enumerate(history)
    )

    prompt = PROFILE_PROMPT_TEMPLATE.format(history=history_text)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3,
                max_tokens=350,
            )
            profile = resp.choices[0].message.content.strip()
            if 'profiles' not in cache:
                cache['profiles'] = {}
            cache['profiles'][uid] = profile
            return profile
        except Exception as e:
            wait = 2 ** attempt
            print(f"    Profile API error: {e}. Retry in {wait}s ...")
            time.sleep(wait)

    return "User enjoys diverse entertainment content."


# ---------------------------------------------------------------------------
# Stage 2: Embedding and cosine similarity ranking
# ---------------------------------------------------------------------------

def get_embedding(client, text, cache, cache_key):
    if cache_key in cache.get('embeddings', {}):
        return np.array(cache['embeddings'][cache_key])

    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=text)
            emb = resp.data[0].embedding
            if 'embeddings' not in cache:
                cache['embeddings'] = {}
            cache['embeddings'][cache_key] = emb
            return np.array(emb)
        except Exception as e:
            wait = 2 ** attempt
            print(f"    Embedding API error: {e}. Retry in {wait}s ...")
            time.sleep(wait)

    return np.zeros(1536)


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def llm_scores(profile_emb, candidate_asins, meta_books, client, cache):
    """Return dict {asin: cosine_sim} for all candidates."""
    scores = {}
    for asin in candidate_asins:
        item_emb = get_embedding(client, book_text_for_embedding(asin, meta_books),
                                 cache, cache_key=f'item_{asin}')
        scores[asin] = cosine_sim(profile_emb, item_emb)
    return scores


def bpr_scores(uid, candidate_asins, books_user2idx, books_user_factors,
               books_item2idx, books_item_factors):
    """Return dict {asin: dot_product} or None if user has no BPR embedding."""
    if uid not in books_user2idx:
        return None
    u_idx = books_user2idx[uid]
    if u_idx >= len(books_user_factors):
        return None
    u_emb = books_user_factors[u_idx]
    scores = {}
    for asin in candidate_asins:
        if asin in books_item2idx:
            i_idx = books_item2idx[asin]
            if i_idx < len(books_item_factors):
                scores[asin] = float(np.dot(u_emb, books_item_factors[i_idx]))
            else:
                scores[asin] = 0.0
        else:
            scores[asin] = 0.0
    return scores


def minmax_norm(score_dict):
    """Normalise score dict values to [0, 1]."""
    vals = list(score_dict.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def rank_candidates(profile_emb, candidate_asins, meta_books, client, cache,
                    uid=None, books_user2idx=None, books_user_factors=None,
                    books_item2idx=None, books_item_factors=None,
                    collab_weight=0.0):
    """
    Rank candidates by blended score:
      final_score = (1 - collab_weight) * llm_cosine + collab_weight * bpr_dot
    Both score sets are min-max normalised before blending.
    If collab_weight == 0 or BPR embedding unavailable, uses LLM scores only.
    """
    lscores = llm_scores(profile_emb, candidate_asins, meta_books, client, cache)

    if collab_weight > 0 and uid is not None and books_user2idx is not None:
        bscores = bpr_scores(uid, candidate_asins, books_user2idx, books_user_factors,
                             books_item2idx, books_item_factors)
        if bscores is not None:
            lscores_n = minmax_norm(lscores)
            bscores_n = minmax_norm(bscores)
            final = {a: (1 - collab_weight) * lscores_n[a] + collab_weight * bscores_n[a]
                     for a in candidate_asins}
            return sorted(candidate_asins, key=lambda a: final[a], reverse=True)

    return sorted(candidate_asins, key=lambda a: lscores[a], reverse=True)


# ---------------------------------------------------------------------------
# Main per-split loop
# ---------------------------------------------------------------------------

def run_split(split_name, users, all_book_asins, meta_movies, meta_books,
              client, books_user2idx, books_user_factors,
              books_item2idx=None, books_item_factors=None):
    print(f"\n--- Split: {split_name} ({len(users):,} users) ---")
    is_sparse = (split_name == 'sparse')

    cache_path = os.path.join(OUT_DIR, f'{split_name}_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        n_profiles = len(cache.get('profiles', {}))
        n_embs     = len(cache.get('embeddings', {}))
        print(f"  Cache loaded: {n_profiles} profiles, {n_embs} embeddings")
    else:
        cache = {}

    rng = random.Random(SEED)

    eval_users = users
    if N_EVAL_USERS > 0 and len(users) > N_EVAL_USERS:
        eval_users = rng.sample(users, N_EVAL_USERS)
        print(f"  Sampled {N_EVAL_USERS} users from {len(users):,}")

    rankings   = {}
    candidates_out = {}

    for entry in tqdm(eval_users, desc=f"  {split_name}", unit=" users"):
        uid = entry['user_id']

        # Sample candidates
        positive, negatives = sample_candidates(entry, all_book_asins, rng)
        candidates_out[uid] = {'positive': positive, 'negatives': negatives}
        all_cands = [positive] + negatives

        # Stage 1: generate profile
        profile = generate_profile(client, entry, meta_movies, cache)

        # Stage 2: embed profile and rank
        profile_emb = get_embedding(client, profile, cache, cache_key=f'profile_{uid}')

        # Score-level blend for sparse users (avoids dimension mismatch)
        cw = COLLAB_WEIGHT if is_sparse else 0.0
        ranked = rank_candidates(
            profile_emb, all_cands, meta_books, client, cache,
            uid=uid,
            books_user2idx=books_user2idx,
            books_user_factors=books_user_factors,
            books_item2idx=books_item2idx,
            books_item_factors=books_item_factors,
            collab_weight=cw,
        )
        rankings[uid] = ranked

        # Periodic cache save
        if len(rankings) % 50 == 0:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Final save
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_pkl(rankings,       os.path.join(OUT_DIR, f'{split_name}_rankings.pkl'))
    save_pkl(candidates_out, os.path.join(OUT_DIR, f'{split_name}_candidates.pkl'))
    print(f"  Saved {len(rankings)} rankings")


def main():
    print("=" * 60)
    print("Condition B: LLM Knowledge Transfer Bridge")
    print("=" * 60)

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("\nLoading data ...")
    cold_start_splits   = load_pkl(os.path.join(PROC_DIR, 'cold_start_splits.pkl'))
    meta_movies         = load_pkl(os.path.join(PROC_DIR, 'meta_movies.pkl'))
    meta_books          = load_pkl(os.path.join(PROC_DIR, 'meta_books.pkl'))
    books_5core         = load_pkl(os.path.join(PROC_DIR, 'books_5core.pkl'))
    books_user2idx      = load_pkl(os.path.join(BPR_DIR,  'books_user2idx.pkl'))
    books_item2idx      = load_pkl(os.path.join(BPR_DIR,  'books_item2idx.pkl'))
    books_user_factors  = np.load( os.path.join(BPR_DIR,  'books_user_factors.npy'))
    books_item_factors  = np.load( os.path.join(BPR_DIR,  'books_item_factors.npy'))

    all_book_asins = frozenset(
        asin for v in books_5core.values() for (asin, _, _) in v
    )
    print(f"  Books candidate pool : {len(all_book_asins):,} items")
    print(f"  BPR-MF Books users   : {books_user_factors.shape[0]:,}")

    for split_name, users in cold_start_splits.items():
        run_split(split_name, users, all_book_asins, meta_movies, meta_books,
                  client, books_user2idx, books_user_factors,
                  books_item2idx=books_item2idx,
                  books_item_factors=books_item_factors)

    print("\n" + "=" * 60)
    print("BRIDGE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {OUT_DIR}")
    print("Run evaluate.py to compute NDCG@10, HR@10, MRR.")


main()
