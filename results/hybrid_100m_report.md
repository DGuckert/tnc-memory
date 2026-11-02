# 100M-token scale test: lexical/hybrid locator

**Store:** 781,250 chunks / 100,000,000 content tokens  
**Queries:** 50 (first 50 of 200 facts)  
**Construction:** 9.75M corpus repeated to 100M; 200 facts + 600 distractors injected once  
**Total storage:** **43.63 B/token** — under the 64 B/token budget

## Results

| retriever | exact r@1 | exact r@10 | exact r@50 | typo r@1 | typo r@10 | typo r@50 |
|---|---:|---:|---:|---:|---:|---:|
| word BM25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.140 | 0.220 |
| char 2-3gram | 0.640 | 0.720 | 0.840 | 0.460 | 0.560 | 0.700 |
| router | 1.000 | 1.000 | 1.000 | 0.460 | 0.560 | 0.700 |

- **Exact-name path is perfect at 100M:** word BM25 r@1/r@10/r@50 = **1.000/1.000/1.000**.
- **Typo fallback:** char n-gram r@10 **0.560**; router gets the same typo quality and the word path's exact quality.
- **Same-noun top-1:** word exact **0.000**; char typo **0.280**.

## Storage

| component | MB | B/token |
|---|---:|---:|
| word BM25 | 282.4 | 2.82 |
| char 2-3gram TF-IDF | 3680.2 | 36.80 |
| **locator combined** | **3962.6** | **39.63** |
| base token IDs + blank flag | 400.0 | 4.00 |
| **total** | **4362.6** | **43.63** |
| + dense bge-small | 700.0 | 7.00 |
| **total with dense** | **5062.6** | **50.63** |
| budget | | 64.00 |

The word BM25 estimate is `postings × 5 B + vocabulary × 24 B`; the char estimate is `nnz × 8 B + 2^20 × 4 B`. Both are in-memory estimates, not compressed on-disk sizes. The 100M run used **459,496,206 char nnz** and **55,912,860 word postings**.

## Build and query time

- Word BM25 build: **78.9 s** (cold run).
- Char TF-IDF build: **276.9 s** (cold run).
- Exact word query: **16.1 ms** median, **21.5 ms** p95 over 50 queries.
- Char fallback: **~2.36 s/query** (235.8 s for 50 exact + 50 typo char searches; the char search dominates the 4-search-per-query aggregate).

The exact path is cheap; the character fallback is the latency problem at 100M. It needs an inverted character index or a smaller/quantized n-gram representation before production.

## Scale comparison

| store | word exact r@10 | char typo r@10 | total B/token |
|---|---:|---:|---:|
| 9.75M | 0.995 | 0.650 | 44.32 |
| 100M | 1.000 | 0.560 | 43.63 |

Quality is flat for exact names and drops modestly for typos; storage is essentially flat. The 100M store is repeated filler, so this is a scale/memory test, not a test of 100M unique content.

## Caveats

- **Repeated corpus.** 9.75M unique tokens repeated 10.26×; facts are injected once. The index size and latency are realistic, but the content diversity is not.
- **Synthetic queries.** Exact project names and template questions; no pronouns or paraphrases.
- **50 queries.** The 95% CI is roughly ±0.1; exact 50/50 is 1.000 but the true rate could be lower.
- **Memory estimates.** Python `postings`/CSR overhead can exceed the estimates; the run used ~25GB RAM and completed without OOM.
- **Char fallback latency.** ~2.4s/query is not production-ready; optimize before shipping.

## Verdict

**The idea scales to 100M tokens within 64 B/token.** Exact-name retrieval stays at **1.000 r@10** at **43.63 B/token**; typo fallback is **0.560 r@10** but costs seconds per query. The next engineering task is the character index, not the memory budget.

## Files

- `hybrid_results_100m.json` — 100M results.
- `hybrid_100m_report.md` — this report.
- `needles_100000000.json` — 100M fact placements.
- `mod_texts_100000000.jsonl` — injected fact texts.