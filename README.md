# Cross-Domain LLM Recommendation: Solving Cold-Start with Language Models

**Research paper:** *Bridging the Gap: Using Large Language Models to Solve Cold-Start Problems in Cross-Domain Recommendation*  
**Author:** Soumyabrata Bairagi &nbsp;·&nbsp; MSc Data Science for Business, HEC Paris &nbsp;·&nbsp; 2026

---

## Overview

This repository contains the implementation code for a study comparing two ways of using an AI language model for cross-domain recommendation — specifically, predicting book recommendations from a user's movie and TV watch history.

The central question: **does a two-step "LLM Bridge" pipeline outperform a simpler one-step "Zero-Shot" approach, and does the answer depend on how little Books history the user has?**

---

## Research Question

> Are AI language models more effective when used as plain **zero-shot** recommenders, or when used as **knowledge-transfer bridges** — and does the answer change depending on how sparse a user's target-domain history is?

Three cold-start severity levels are tested separately:

| Group | Books interactions | Notes |
|---|---|---|
| `pure` | 0 | Correct answer is random — calibration check only |
| `very_cold` | 1–2 | Thin real signal |
| `sparse` | 3–5 | Usable real signal |

---

## Key Findings

- **Zero-Shot = Bridge** in `very_cold` (p = 0.82, no significant difference)
- **Bridge > Zero-Shot** in `sparse` (p = 0.02, NDCG@10 +0.041) — the crossover is statistically confirmed
- A simple **Hybrid popularity blend** (no retraining) lifts both AI approaches significantly in `very_cold` and `sparse`
- The Bridge advantage in `sparse` is driven by its BPR-MF collaborative-filtering blend, not the profile-writing step

---

## Methods Implemented

| Method | Type |
|---|---|
| Zero-Shot LLM | AI — direct ranking from history |
| LLM Bridge | AI — taste profile → cosine similarity; + BPR-MF blend for sparse |
| BPR-MF | Classical — matrix factorisation, no cross-domain transfer |
| EMCDR | Classical — embedding + mapping across domains |
| User-CF | Classical — similar-user cross-domain transfer |
| Random | Control baseline |
| Popularity | Control — always recommends most popular |
| Anti-Popularity | Control — always recommends least popular |
| Hybrid (Zero-Shot) | Add-on — adaptive α blend of Zero-Shot + popularity |
| Hybrid (Bridge) | Add-on — adaptive α blend of Bridge + popularity |

---

## Repository Structure

```
.
├── CDR_v3.ipynb          # Main notebook — full pipeline end to end
├── requirements.txt      # Python dependencies
├── README.md
└── data/                 # (not included — see Data section below)
    ├── Movies_and_TV.jsonl
    └── Books.jsonl
```

---

## Setup

**Python 3.9+ recommended.**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Key dependencies:

```
openai
numpy
pandas
scikit-learn
torch
scipy
tqdm
```

You will also need an **OpenAI API key** for the LLM-based methods (Zero-Shot, Bridge). Set it as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

The LLM Bridge uses `text-embedding-3-small` (1,536 dimensions) for embedding profiles and candidate book descriptions.

---

## Data

This project uses the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset (McAuley Lab, UCSD), specifically:

- `Movies_and_TV` — source domain
- `Books` — target domain

Download the raw `.jsonl` files from the [Hugging Face Datasets page](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and place them in the `data/` directory.

> **Note:** The Books file is large. The notebook caps it at 20,000,000 lines for preprocessing time and memory reasons. Full preprocessing details are documented in the notebook.

---

## Running the Notebook

Open `CDR_v3.ipynb` in Jupyter and run sections in order:

1. **Data preprocessing** — k-core filtering, overlap user identification, cold-start splits
2. **Method implementations** — each method has its own section
3. **Evaluation** — NDCG@10, HR@10, MRR across all methods and groups
4. **Statistical tests** — crossover regression (headline test) and Random-baseline comparisons

> The Zero-Shot and LLM Bridge methods make OpenAI API calls and will incur costs. Bridge for the `sparse` group is the most expensive (~1hr 20min runtime due to the BPR-MF blending step; consider pre-caching embeddings for production use).

---

## Results Summary

**NDCG@10 by method and cold-start group** (500 users per group):

| Method | pure | very_cold | sparse |
|---|---|---|---|
| Zero-Shot LLM | 0.2185 | 0.2298 | 0.2272 |
| LLM Bridge | 0.2078 | 0.2154 | **0.2571** |
| BPR-MF | 0.2300 | 0.2300 | 0.2657 |
| EMCDR | 0.2235 | 0.2629 | 0.2569 |
| Random (baseline) | 0.2300 | 0.2427 | 0.2144 |
| Hybrid Zero-Shot | — | 0.3474 | 0.2883 |
| Hybrid Bridge | — | 0.3451 | **0.3140** |

> Popularity (0.365 in `very_cold`) and Anti-Popularity (0.434) are excluded from this summary — they are measurement artifacts of the 20-candidate sampling setup, not genuine competitors. See paper §2.3 and §4.1.2.

---

## Statistical Test

The headline test uses a crossover-design regression — an interaction term between method type (zero-shot vs bridge) and cold-start group — to directly answer the research question:

| Group | NDCG@10 change | p-value | Interpretation |
|---|---|---|---|
| very_cold | −0.004 | 0.82 | No significant difference |
| sparse | +0.041 | **0.02** | Statistically significant |

---

## Citation

If you use this code or build on this work, please cite:

```
Soumyabrata Bairagi (2026). Bridging the Gap: Using Large Language Models to Solve 
Cold-Start Problems in Cross-Domain Recommendation. 
MSc Research Paper, HEC Paris.
```

---

## License

This repository is shared for academic and research purposes.  
Please credit the author if you use or adapt any part of this code.

---

## Acknowledgements

Data: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), McAuley Lab, UC San Diego.  
LLM inference and embeddings: OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`).
