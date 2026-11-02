# Cross-encoder reranker v0.1

Fine-tuned `cross-encoder/ms-marco-MiniLM-L-6-v2` on the hard-negative reranker set.

- **Base:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Training:** 5,000 queries / 29,991 pairs, 1 epoch, batch 32, 938 steps
- **Time:** 119 seconds on the RTX 2060
- **Output:** `v0.1-reranker` release asset

## Pairwise validation

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Top-1 | 1.000 | 1.000 |
| MRR | 1.000 | 1.000 |
| Mean score margin | 4.32 | 17.94 |

The synthetic pairwise set is easy: the positive fact contains the exact project name and code. The reranker already solves it before fine-tuning; the fine-tune mainly increases the margin.

## End-to-end retrieval at 9.75M tokens

The reranker scores the top 50 candidates from word BM25 (exact query) or character n-gram search (one-character typo).

| Query | Retriever | r@1 | r@10 | r@50 |
|---|---|---:|---:|---:|
| Exact | word pre-rerank | 0.980 | 0.995 | 1.000 |
| Exact | + base reranker | 0.965 | 1.000 | 1.000 |
| Exact | + fine-tuned reranker | **0.995** | **1.000** | **1.000** |
| Typo | char pre-rerank | 0.430 | 0.650 | 0.805 |
| Typo | + base reranker | 0.665 | 0.780 | 0.805 |
| Typo | + fine-tuned reranker | **0.680** | 0.775 | 0.805 |

- Exact top-1 improves from **0.980** to **0.995**.
- Typo top-1 improves from **0.430** to **0.680**.
- Typo top-10 improves from **0.650** to **0.775–0.780**.
- The candidate ceiling is **1.000** for exact and **0.805** for typo; the reranker cannot recover gold outside the top 50.

## Where it helps

- The cross-encoder reads the whole chunk, including the appended fact, so it can prefer the note with the matched project over same-noun chatter.
- For typos, the character retriever supplies the candidate; the reranker then picks the right note from the noisy top 50.

## Caveats

- The pairwise validation is too easy; the end-to-end number is the meaningful one.
- The fine-tune improves top-1 and MRR; the typo top-10 difference from the base is within noise.
- The 9.75M benchmark is synthetic and template-based.
- Reranking adds query latency; benchmark before production.

## Files

- `reranker_metrics.json` — pairwise validation.
- `reranker_975m.json` — end-to-end retrieval.
- `reranker-mini` release asset — fine-tuned cross-encoder.
