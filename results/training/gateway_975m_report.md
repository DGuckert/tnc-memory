# Gateway at 9.75M tokens

Full pipeline on the 9.75M-token store:

```
conversation
  -> explicit project shortcut or middleman tool call
  -> entity correction + word BM25 / character fallback
  -> canonical-query reranker top 5
  -> middleman reader
  -> answer
```

## Results

200 conversations: 100 pronoun, 50 explicit, 50 typo.

| Case | Search rate | Correct project | Correct answer |
|---|---:|---:|---:|
| Pronoun (“it”) | 1.00 | 0.99 | **0.97** |
| Explicit name | 1.00 | 1.00 | **0.94** |
| One-char typo | 1.00 | 0.96 | **0.94** |
| **Overall** | 1.00 | — | **0.955** |

- The explicit-project shortcut bypasses the middleman when the user names a known project. A context shortcut also resolves a pronoun when the conversation contains exactly one known project.
- Pronoun resolution is strong: **99%** correct project, **97%** correct answer.
- Typo correction is strong: **96%** correct project, **94%** correct answer.
- The remaining errors are reader mistakes, not retrieval failures.

## Latency

| Stage | Median | p95 |
|---|---:|---:|
| Middleman tool call | 1.91 s | 15.64 s |
| Retrieval | 0.12 s | 0.54 s |
| Rerank | 0.08 s | 0.13 s |
| Reader | 0.60 s | 4.04 s |
| **Total** | **2.62 s** | **20.07 s** |

Previous latency was **4.35 s** total and **1.70 s** reader. The shortcuts and shorter generation cut the median total by about **40%** and the reader by about **65%**. The tool-call median mixes one-project pronoun cases that still use the model with shortcut cases that do not.

## Memory and build

- Index build: **51 s** (warm run; cold build was ~117 s).
- GPU peak: **1.04 GB**.
- Process RSS: **2.89 GB**.
- Store: 76,145 chunks / 9,746,560 content tokens.

## Caveats

- The benchmark facts and queries are synthetic; the 200 facts are the same benchmark used throughout.
- Only 200 conversations.
- The latency was measured on a shared GPU; absolute numbers will vary.
- The middleman and reader share one 360M model.
- The reader makes 3–6% errors even when retrieval is correct.
