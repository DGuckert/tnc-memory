# TNC Memory

Experiments on compact long-term memory for chat. The goal is to keep a large history in a small token budget, find the right note, and let a model answer from it.

## Main result

A dense-only locator was not good enough: **0.405 recall@10** at 9.75M tokens. Adding exact-word search and a spelling fallback fixed the main failure:

| Store | Exact-name r@10 | One-char typo r@10 | Total storage |
|---|---:|---:|---:|
| 9.75M | **0.995** | **0.650** | 44.32 B/token |
| 100M | **1.000** (50 q) | **0.560** (50 q) | 43.63 B/token |

The 100M store was built by repeating the 9.75M corpus to 781,250 chunks and injecting 800 synthetic facts once. It is a scale/memory test, not 100M unique conversations.

## What worked

- **Word BM25** for exact project names: 0.995 r@10 at 9.75M; 1.000 at 100M.
- **Character 2-3-gram search** for spelling variants: 0.650 r@10 at 9.75M; 0.560 at 100M.
- **Router**: use word search when the entity name is present, character search when it is not.
- **Storage**: 44.32 B/token at 9.75M and 43.63 B/token at 100M, under the 64 B/token budget.

## What did not work

- Dense-only retrieval: **0.405** r@10 at 9.75M; **79.5%** of first hits were same-noun/wrong-project.
- PQ compression: 48-byte codes got **0.235** r@10; 8-byte codes got **0.010**.
- Mean-pooled token windows: **0.020** r@10, median gold rank 5,535.
- A 360M LLM reader: **0.40** at 2K and **0.20** at 4K; tags did not help.
- No-name questions: word BM25 falls to **0.045** r@10. The query is ambiguous; conversation state/entity resolution is required.

## Repo layout

- `scripts/` — experiment and report-generation code.
- `results/` — curated reports, JSON results, and plots.
- `results/report.md` — dense locator study.
- `results/llm_report.md` — small-LLM reader study.
- `results/hybrid_report.md` — 9.75M lexical/hybrid result.
- `results/hybrid_100m_report.md` — 100M scale result.
- `training/` — synthetic SFT and reranker data; see `training/README.md`.

## Training data

`training/` contains 10,000 train SFT rows, 1,500 validation rows, 1,500 test rows, and matching reranker pairs. The SFT set mixes two tasks:

- **Tool call:** conversation → `search_memory`, `ask_clarification`, or `no_tool`.
- **Answer from memory:** notes + question → exact code or `I don't know.`

Project names and codes are disjoint across splits, and the 200 benchmark facts are excluded. The data is synthetic; raw Discord text is not included.

## Data note

The raw Discord corpus, chunk text, embeddings, and large arrays are **not** committed. Only curated results, scripts, and synthetic training data are included. The facts, project names, and codes are synthetic. This repository is private.

## Reproduce

The scripts expect a Linux/WSL workspace at `/home/daniel/tnc` with the prepared corpus and `work/` data. Install dependencies with `pip install -r requirements.txt`; install the CUDA build of PyTorch separately if using a GPU. The report generators read the curated JSON files in `work/`.

## Status

- Retrieval is strong for questions that name the entity.
- Typos are handled but the character fallback is slow at 100M (~2.36 s/query).
- The next step is a memory gateway (middleman) that resolves “it/that project,” rewrites the query, retrieves, and hands notes to any host model.
- The training set for the middleman and reader is ready; the next step is a small SFT run and a cross-encoder reranker.
- The 360M reader is too weak; use a 0.5B–1B reader or the host model after retrieval.
