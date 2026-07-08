"""
preprocess.py
=============
Cross-domain recommendation preprocessing pipeline.
Source domain : Amazon Movies & TV
Target domain : Amazon Books

Hardware notes (CPU-only, no GPU required):
  - Streaming design: peak RAM ~2-3 GB even for the full 6+ GB compressed Books file.
  - Estimated runtime on a typical laptop (SSD): ~20-30 min for full data.
  - For a quick dry run, set SAMPLE = 200_000 in the Configuration block below.
  - No GPU required. BPR-MF uses the 'implicit' library (CPU+BLAS).

Outputs saved to <DATA_DIR>/processed/:
  interactions_movies.pkl  - {user_id: [(parent_asin, timestamp, rating), ...]} sorted by time
  interactions_books.pkl   - same for Books
  movies_5core.pkl         - {user_id: [...]} after iterative 5-core filtering
  books_5core.pkl          - same for Books
  overlap_users.pkl        - set of user_ids present in both 5-core sets
  cold_start_splits.pkl    - dict with keys 'pure', 'very_cold', 'sparse'
                             each a list of user dicts with fields:
                             user_id, movies_interactions, n_books,
                             books_train, books_val, books_test
  meta_movies.pkl          - {parent_asin: {title, description, categories, features}}
  meta_books.pkl           - same for Books (includes 'author' field)

Usage in Jupyter:
  Set DATA_DIR / SAMPLE in the Configuration block, then run the script.
  The script does NOT use argparse, so it works in notebooks without errors.
"""

import gzip
import json
import pickle
import os
from collections import defaultdict
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Step 1: Load interactions
# ---------------------------------------------------------------------------

def load_interactions(path, domain_name, max_lines=0):
    """
    Stream JSONL.gz and build user2items dict.
    Deduplicates (user, parent_asin) -- keeps highest-rated, then latest.

    Memory design: uses a single nested dict {uid: {asin: (ts, rating)}} during
    streaming (one dict, not two), then converts in-place via popitem() so the
    source dict shrinks as the output dict grows -- peak RAM is ~1x not ~2x.
    """
    print(f"\n[1] Loading {domain_name} from {os.path.basename(path)} ...")
    # user2best: {uid: {asin: (ts, rating)}} -- one dict, organised by user
    user2best = defaultdict(dict)
    total = 0

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"  {domain_name}", unit=" lines", mininterval=2.0)):
            if max_lines > 0 and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = rec.get('user_id')
            asin    = rec.get('parent_asin') or rec.get('asin')
            ts      = rec.get('timestamp', 0)
            rating  = rec.get('rating', 0.0)

            if not user_id or not asin:
                continue

            best = user2best[user_id].get(asin)
            if best is None or (rating, ts) > (best[1], best[0]):
                user2best[user_id][asin] = (ts, rating)
            total += 1

    n_users = len(user2best)
    n_pairs = sum(len(v) for v in user2best.values())
    n_items = len({a for v in user2best.values() for a in v})
    print(f"  Raw records           : {total:,}")
    print(f"  Unique (user,item)    : {n_pairs:,}")
    print(f"  Unique users          : {n_users:,}")
    print(f"  Unique items          : {n_items:,}")

    # Convert in-place: popitem() frees user2best as user2items grows
    # Peak RAM stays at ~1x the data size, not ~2x
    user2items = {}
    while user2best:
        uid, item_dict = user2best.popitem()
        user2items[uid] = sorted(
            [(asin, ts, r) for asin, (ts, r) in item_dict.items()],
            key=lambda x: x[1]
        )

    return user2items


# ---------------------------------------------------------------------------
# Step 2: Iterative k-core filtering
# ---------------------------------------------------------------------------

def kcore_filter(user2items, k=5, domain_name='domain'):
    """Iteratively remove users/items with fewer than k interactions."""
    print(f"\n[2] {k}-core filtering on {domain_name} ...")
    data = {uid: list(v) for uid, v in user2items.items()}
    iteration = 0

    while True:
        iteration += 1
        item_count = defaultdict(int)
        for interactions in data.values():
            for (asin, ts, rating) in interactions:
                item_count[asin] += 1

        valid_items = {asin for asin, cnt in item_count.items() if cnt >= k}

        new_data = {}
        for uid, interactions in data.items():
            filtered = [(a, t, r) for (a, t, r) in interactions if a in valid_items]
            if len(filtered) >= k:
                new_data[uid] = filtered

        removed = len(data) - len(new_data)
        if removed == 0:
            break
        data = new_data
        print(f"  Iter {iteration}: removed {removed:,} users -> {len(data):,} remaining")

    n_int   = sum(len(v) for v in data.values())
    n_items = len({a for v in data.values() for (a, _, _) in v})
    print(f"  Final: {len(data):,} users, {n_items:,} items, {n_int:,} interactions")
    return data


# ---------------------------------------------------------------------------
# Step 3: Overlapping users
# ---------------------------------------------------------------------------

def find_overlap(movies_5core, books_5core):
    overlap = set(movies_5core.keys()) & set(books_5core.keys())
    print(f"\n[3] Overlapping users: {len(overlap):,}")
    print(f"  {100*len(overlap)/max(len(movies_5core),1):.1f}% of Movies 5-core")
    print(f"  {100*len(overlap)/max(len(books_5core),1):.1f}% of Books 5-core")
    return overlap


# ---------------------------------------------------------------------------
# Step 4: Cold-start splits
# ---------------------------------------------------------------------------

def build_cold_start_splits(movies_raw, books_raw, movies_5core):
    """
    Classify movies_5core users by target-domain (Books) interaction count:
      pure       : 0 Books interactions
      very_cold  : 1-2 Books interactions
      sparse     : 3-5 Books interactions
    Users with >5 Books interactions are excluded (not cold-start).
    Leave-one-out: last interaction = test, second-to-last = val, rest = train.
    """
    print("\n[4] Building cold-start splits ...")

    splits = {'pure': [], 'very_cold': [], 'sparse': []}

    for uid, movie_ints in movies_5core.items():
        book_ints = sorted(books_raw.get(uid, []), key=lambda x: x[1])
        n = len(book_ints)

        entry = {
            'user_id':             uid,
            'movies_interactions': movie_ints,
            'n_books':             n,
            'books_train':         [],
            'books_val':           None,
            'books_test':          None,
        }

        if n == 0:
            splits['pure'].append(entry)
        elif n <= 2:
            entry['books_test']  = book_ints[-1]
            entry['books_val']   = book_ints[-2] if n == 2 else None
            entry['books_train'] = book_ints[:-1] if n == 2 else []
            splits['very_cold'].append(entry)
        elif n <= 5:
            entry['books_test']  = book_ints[-1]
            entry['books_val']   = book_ints[-2] if n >= 2 else None
            entry['books_train'] = book_ints[:-2] if n >= 3 else []
            splits['sparse'].append(entry)
        # n > 5: skip (warm user)

    for key, lst in splits.items():
        print(f"  {key:12s}: {len(lst):,} users")

    return splits


# ---------------------------------------------------------------------------
# Step 5: Load metadata
# ---------------------------------------------------------------------------

def load_metadata(meta_path, relevant_asins, domain_name):
    """Stream metadata JSONL.gz, keep only items in relevant_asins."""
    print(f"\n[5] Loading {domain_name} metadata ({len(relevant_asins):,} relevant items) ...")
    meta = {}

    with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"  {domain_name} meta", unit=" lines", mininterval=2.0):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = rec.get('parent_asin')
            if asin not in relevant_asins:
                continue

            desc = rec.get("description") or []
            if isinstance(desc, list):
                desc = ' '.join(d for d in desc if d)
            desc = desc.strip()

            entry = {
                'title':       (rec.get('title') or '').strip(),
                'description': desc,
                'categories':  rec.get('categories') or [],
                'features':    rec.get('features') or [],
            }

            author = rec.get('author')
            if isinstance(author, dict):
                entry['author'] = author.get('name', '')
            elif isinstance(author, str):
                entry['author'] = author

            meta[asin] = entry

    pct = 100 * len(meta) / max(len(relevant_asins), 1)
    print(f"  Loaded {len(meta):,} items ({pct:.1f}% coverage)")
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # =========================================================================
    # CONFIGURATION -- edit these values as needed
    DATA_DIR = r'D:\Study\HEC Paris\Courses\Research\Research Thesis\Data'
    K        = 5             # k-core threshold (standard: 5)
    SAMPLE   = 20_000_000   # 0 = full data; e.g. 200_000 for a quick test run
    # =========================================================================

    # Allow override via environment variables when running from terminal
    DATA_DIR = os.environ.get('DATA_DIR', DATA_DIR)
    K        = int(os.environ.get('K', K))
    SAMPLE   = int(os.environ.get('SAMPLE', SAMPLE))

    out_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    if SAMPLE > 0:
        print(f"\n*** SAMPLE MODE: {SAMPLE:,} lines per file ***")
        print("*** Set SAMPLE = 0 for full data ***\n")

    movies_path      = os.path.join(DATA_DIR, 'Movies_and_TV.jsonl.gz')
    books_path       = os.path.join(DATA_DIR, 'Books.jsonl.gz')
    meta_movies_path = os.path.join(DATA_DIR, 'meta_Movies_and_TV.jsonl.gz')
    meta_books_path  = os.path.join(DATA_DIR, 'meta_Books.jsonl.gz')

    # Step 1 -- load interactions one domain at a time, free after use
    movies_raw = load_interactions(movies_path, 'Movies & TV', max_lines=SAMPLE)
    save(movies_raw, os.path.join(out_dir, 'interactions_movies.pkl'))

    books_raw = load_interactions(books_path, 'Books', max_lines=SAMPLE)
    save(books_raw, os.path.join(out_dir, 'interactions_books.pkl'))

    # Step 2 -- filter each domain, then free raw data
    movies_5core = kcore_filter(movies_raw, k=K, domain_name='Movies & TV')
    save(movies_5core, os.path.join(out_dir, 'movies_5core.pkl'))

    books_5core = kcore_filter(books_raw, k=K, domain_name='Books')
    save(books_5core, os.path.join(out_dir, 'books_5core.pkl'))

    # Step 3
    overlap = find_overlap(movies_5core, books_5core)
    save(overlap, os.path.join(out_dir, 'overlap_users.pkl'))

    # Step 4 -- build cold-start splits, then free large raw dicts
    cold_start_splits = build_cold_start_splits(movies_raw, books_raw, movies_5core)
    save(cold_start_splits, os.path.join(out_dir, 'cold_start_splits.pkl'))
    del movies_raw, books_raw   # free ~2-4 GB before metadata loading

    # Step 5 -- collect relevant item ASINs first
    movies_items = {a for v in movies_5core.values() for (a, _, _) in v}
    books_items  = {a for v in books_5core.values() for (a, _, _) in v}
    for split_users in cold_start_splits.values():
        for entry in split_users:
            for (a, _, _) in (entry.get('books_train') or []):
                books_items.add(a)
            if entry.get('books_val'):
                books_items.add(entry['books_val'][0])
            if entry.get('books_test'):
                books_items.add(entry['books_test'][0])

    print(f"\n  Relevant Movies items : {len(movies_items):,}")
    print(f"  Relevant Books items  : {len(books_items):,}")

    meta_movies = load_metadata(meta_movies_path, movies_items, 'Movies & TV')
    save(meta_movies, os.path.join(out_dir, 'meta_movies.pkl'))

    meta_books = load_metadata(meta_books_path, books_items, 'Books')
    save(meta_books, os.path.join(out_dir, 'meta_books.pkl'))

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Movies 5-core : {len(movies_5core):,} users")
    print(f"Books 5-core  : {len(books_5core):,} users")
    print(f"Overlap users : {len(overlap):,}")
    for key, lst in cold_start_splits.items():
        print(f"  {key:12s}: {len(lst):,} users")
    print(f"Movies metadata : {len(meta_movies):,} / {len(movies_items):,} items")
    print(f"Books metadata  : {len(meta_books):,} / {len(books_items):,} items")
    print(f"\nFiles saved to: {out_dir}")


main()
