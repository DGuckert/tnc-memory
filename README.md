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

## Trained baseline

A LoRA for `SmolLM2-360M-Instruct` was trained for 120 steps on 1,920 examples. On held-out projects (200 tool + 200 reader):

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Valid tool JSON | 0.00 | 1.00 |
| Full tool-call accuracy | 0.00 | 0.97 |
| Reader accuracy | 0.21 | 0.99 |

Harder stress test, 100 conversations with real chatter noise and paraphrases: fine-tuned full tool accuracy **0.86**; base **0.00**.

A cross-encoder reranker is also trained. At 9.75M tokens it improves exact top-1 from **0.980** to **0.995** and typo top-1 from **0.430** to **0.680**.

End-to-end at 9.75M: exact-name questions are **1.000** correct; one-character typos are **0.890** after fuzzy entity correction and canonical-query reranking (char-only baseline: 0.595).

The first full gateway (`scripts/gateway.py`) is integrated. On 100 noisy conversations with real chatter, paraphrases, typos, and pronouns: **0.86** tool-call accuracy and **0.88** end-to-end answer accuracy.

On the 9.75M store, 200 conversations: **0.955** end-to-end answer accuracy (pronoun 0.97, explicit 0.94, typo 0.94). Median latency is **4.35 s**; indices use **1.04 GB GPU** and **2.89 GB RAM**.

Adapters: `v0.1-middleman` and `v0.1-reranker` releases. Details: `results/training/`.

## Data note

The raw Discord corpus, chunk text, embeddings, and large arrays are **not** committed. Only curated results, scripts, and synthetic training data are included. The facts, project names, and codes are synthetic. This repository is private.

## Reproduce

The scripts expect a Linux/WSL workspace at `/home/daniel/tnc` with the prepared corpus and `work/` data. Install dependencies with `pip install -r requirements.txt`; install the CUDA build of PyTorch separately if using a GPU. The report generators read the curated JSON files in `work/`.

## Status

- Retrieval is strong for questions that name the entity.
- Typos are handled but the character fallback is slow at 100M (~2.36 s/query).
- The next step is a memory gateway (middleman) that resolves “it/that project,” rewrites the query, retrieves, and hands notes to any host model.
- A middleman LoRA is trained and released; held-out test is 0.97 full tool-call and 0.99 reader.
- A cross-encoder reranker is trained; exact top-1 is 0.995 and typo top-1 is 0.680 on the 9.75M benchmark.
- End-to-end retrieval + rerank + reader: exact **1.000**, typo **0.890** with entity correction.
- The gateway is integrated; noisy-conversation end-to-end answer accuracy is **0.88**.
- On 9.75M tokens, gateway answer accuracy is **0.955**; median latency **4.35 s**.
- Next: cut tool/reader latency, then human-labeled conversations.
