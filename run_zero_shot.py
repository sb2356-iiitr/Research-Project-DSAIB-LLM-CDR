"""
run_zero_shot.py
================
Condition A: Zero-shot LLM recommender.

For each cold-start evaluation user, the LLM receives their Movies & TV
interaction history and ranks 20 Books candidates (1 held-out positive +
19 randomly sampled negatives) using world knowledge alone.

Prompt design follows LLMRank (Hou et al., 2024):
  1. User history block (most recent 20 movies, newest first)
  2. Bootstrapping reflection: LLM infers user preferences before ranking
  3. Ranking task: rank the 20 Books candidates

All API responses are cached to disk so the script can be safely interrupted
and resumed without re-incurring costs.

Outputs saved to <DATA_DIR>/processed/results/zero_shot/:
  {split}_rankings.pkl   - {user_id: [ranked list of (asin, rank)]}
  {split}_candidates.pkl - {user_id: {positive: asin, negatives: [asin,...]}}

Usage: edit the Configuration block and run.

Requirements:
  pip install openai tqdm
  Set OPENAI_API_KEY in the Configuration block or as an environment variable.
"""

import os
import re
import json
import pickle
import random
import time
from tqdm import tqdm
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR       = r'D:\Study\HEC Paris\Courses\Research\Research Thesis\Data'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'sk-...')  # replace or set env var
MODEL          = 'gpt-4o-mini'    # cheapest capable model; swap to 'gpt-4o' for quality
MAX_HISTORY    = 20               # max source-domain items in prompt
N_CANDIDATES   = 20               # 1 positive + 19 negatives
N_EVAL_USERS   = 500              # users per sparsity bucket (set to 0 = all)
SEED           = 42
# ---------------------------------------------------------------------------

random.seed(SEED)
PROC_DIR = os.path.join(DATA_DIR, 'processed')
OUT_DIR  = os.path.join(PROC_DIR, 'results', 'zero_shot')
os.makedirs(OUT_DIR, exist_ok=True)


def load_pkl(name):
    with open(os.path.join(PROC_DIR, name), 'rb') as f:
        return pickle.load(f)

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Candidate sampling
# ---------------------------------------------------------------------------

def sample_candidates(user_entry, all_book_asins, rng):
    """
    Returns (positive_asin, [negative_asins]) for a user.
    For pure cold-start users (no test item), sample a random positive.
    """
    # Items the user has already interacted with -- exclude from negatives
    seen_books = {entry[0] for entry in (user_entry.get('books_train') or [])}
    if user_entry.get('books_val'):
        seen_books.add(user_entry['books_val'][0])

    if user_entry.get('books_test'):
        positive = user_entry['books_test'][0]
    else:
        # Pure cold-start: sample a random positive from all books
        candidates_pool = list(all_book_asins - seen_books)
        positive = rng.choice(candidates_pool)

    seen_books.add(positive)
    neg_pool = list(all_book_asins - seen_books)
    negatives = rng.sample(neg_pool, min(N_CANDIDATES - 1, len(neg_pool)))
    return positive, negatives


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_item_text(asin, meta, domain='books'):
    """Build a readable item string from metadata."""
    m = meta.get(asin, {})
    title = m.get('title') or asin
    if domain == 'movies':
        cats = m.get('categories', [])
        genre = ', '.join(cats[:3]) if cats else ''
        feats = m.get('features', [])
        # Features often contain year, IMDB rating etc.
        extra = '; '.join(str(f) for f in feats[:3]) if feats else ''
        parts = [title]
        if genre:
            parts.append(f"[{genre}]")
        if extra:
            parts.append(f"({extra})")
        return ' '.join(parts)
    else:
        m_author = m.get('author', '')
        parts = [title]
        if m_author:
            parts.append(f"by {m_author}")
        cats = m.get('categories', [])
        genre = ', '.join(cats[1:3]) if len(cats) > 1 else ''
        if genre:
            parts.append(f"[{genre}]")
        return ' | '.join(parts)


def build_prompt(user_entry, positive, negatives, meta_movies, meta_books):
    """Build the full ranking prompt for a user."""
    # History: most recent first, up to MAX_HISTORY
    history = list(reversed(user_entry['movies_interactions']))[:MAX_HISTORY]
    history_text = '\n'.join(
        f"  {i+1}. {build_item_text(asin, meta_movies, 'movies')}"
        for i, (asin, ts, rating) in enumerate(history)
    )

    # Shuffle candidates so positive is not always first
    all_candidates = [positive] + negatives
    random.shuffle(all_candidates)
    candidate_text = '\n'.join(
        f"  {chr(65+i)}. {build_item_text(asin, meta_books, 'books')}"
        for i, asin in enumerate(all_candidates)
    )

    prompt = f"""You are a book recommendation assistant. A user has watched the following movies and TV shows (most recent first):

{history_text}

Step 1 - Understand the user's taste: Based on this watch history, briefly describe what kinds of stories, themes, genres, and styles this user tends to enjoy. Be specific.

Step 2 - Rank the books: Given your understanding of the user's taste, rank the following {len(all_candidates)} books from most to least likely to be enjoyed by this user. Output ONLY a ranked list as letters, like: A > C > B > D > ...

Books to rank:
{candidate_text}

Your response:
Step 1 - User taste analysis:
[your analysis here]

Step 2 - Ranked list (letters only, separated by >):
"""
    return prompt, all_candidates


# ---------------------------------------------------------------------------
# LLM call with retry and caching
# ---------------------------------------------------------------------------

def parse_ranking(response_text, n_candidates):
    """
    Extract ranked letter sequence from LLM output.
    Returns list of 0-indexed positions, e.g. [0, 2, 1, ...] meaning A=rank1, C=rank2, B=rank3
    """
    # Find the line containing the ranking (letters separated by >)
    lines = response_text.strip().split('\n')
    ranking_line = ''
    for line in reversed(lines):
        if '>' in line and any(c.isalpha() for c in line):
            ranking_line = line
            break

    if not ranking_line:
        return None

    # Extract letters
    letters = re.findall(r'\b([A-T])\b', ranking_line.upper())
    if not letters:
        return None

    # Convert to 0-based indices
    seen = set()
    indices = []
    for letter in letters:
        idx = ord(letter) - ord('A')
        if 0 <= idx < n_candidates and idx not in seen:
            indices.append(idx)
            seen.add(idx)

    # Fill in any missing candidates at the end
    all_indices = list(range(n_candidates))
    for idx in all_indices:
        if idx not in seen:
            indices.append(idx)

    return indices


def llm_rank(client, prompt, all_candidates, cache, cache_key, retries=3):
    """Call LLM, parse ranking, use cache to avoid re-calling."""
    if cache_key in cache:
        return cache[cache_key]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            text = response.choices[0].message.content
            ranking_indices = parse_ranking(text, len(all_candidates))
            if ranking_indices is None:
                # Fallback: random order
                ranking_indices = list(range(len(all_candidates)))
                random.shuffle(ranking_indices)
            ranked_asins = [all_candidates[i] for i in ranking_indices]
            cache[cache_key] = ranked_asins
            return ranked_asins
        except Exception as e:
            wait = 2 ** attempt
            print(f"    API error (attempt {attempt+1}): {e}. Retrying in {wait}s ...")
            time.sleep(wait)

    # All retries failed: return random ranking
    shuffled = list(all_candidates)
    random.shuffle(shuffled)
    cache[cache_key] = shuffled
    return shuffled


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_split(split_name, users, all_book_asins, meta_movies, meta_books, client):
    print(f"\n--- Split: {split_name} ({len(users):,} users) ---")

    # Load or init cache
    cache_path = os.path.join(OUT_DIR, f'{split_name}_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"  Loaded {len(cache)} cached responses")
    else:
        cache = {}

    rng = random.Random(SEED)

    # Sample users if N_EVAL_USERS is set
    eval_users = users
    if N_EVAL_USERS > 0 and len(users) > N_EVAL_USERS:
        eval_users = rng.sample(users, N_EVAL_USERS)
        print(f"  Sampled {N_EVAL_USERS} users from {len(users):,}")

    rankings   = {}   # {user_id: [ranked asin list]}
    candidates = {}   # {user_id: {positive, negatives}}

    for entry in tqdm(eval_users, desc=f"  {split_name}", unit=" users"):
        uid = entry['user_id']

        # Sample candidates
        positive, negatives = sample_candidates(entry, all_book_asins, rng)
        candidates[uid] = {'positive': positive, 'negatives': negatives}

        # Build prompt and call LLM
        prompt, all_cands = build_prompt(entry, positive, negatives, meta_movies, meta_books)
        ranked = llm_rank(client, prompt, all_cands, cache, cache_key=uid)
        rankings[uid] = ranked

        # Save cache periodically
        if len(rankings) % 50 == 0:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Final cache save
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_pkl(rankings,   os.path.join(OUT_DIR, f'{split_name}_rankings.pkl'))
    save_pkl(candidates, os.path.join(OUT_DIR, f'{split_name}_candidates.pkl'))
    print(f"  Saved {len(rankings)} rankings for split '{split_name}'")


def main():
    print("=" * 60)
    print("Condition A: Zero-Shot LLM Recommender")
    print("=" * 60)

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("\nLoading data ...")
    cold_start_splits = load_pkl('cold_start_splits.pkl')
    meta_movies       = load_pkl('meta_movies.pkl')
    meta_books        = load_pkl('meta_books.pkl')
    books_5core       = load_pkl('books_5core.pkl')

    # All Books items in 5-core (candidate pool)
    all_book_asins = frozenset(
        asin for v in books_5core.values() for (asin, _, _) in v
    )
    print(f"  Books candidate pool : {len(all_book_asins):,} items")

    for split_name, users in cold_start_splits.items():
        run_split(split_name, users, all_book_asins, meta_movies, meta_books, client)

    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {OUT_DIR}")
    print("Run evaluate.py to compute NDCG@10, HR@10, MRR.")


main()
