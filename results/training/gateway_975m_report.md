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
| Pronoun (“it”) | 1.00 | 0.97 | **0.97** |
| Explicit name | 1.00 | 1.00 | **0.94** |
| One-char typo | 1.00 | 0.94 | **0.92** |
| **Overall** | 1.00 | — | **0.950** |

- The explicit-project shortcut bypasses the middleman when the user names a known project. A context shortcut also resolves a pronoun when the conversation contains exactly one known project.
- Pronoun resolution: **97%** correct project, **97%** correct answer.
- Typo correction: **94%** correct project, **92%** correct answer.
- The remaining errors are reader mistakes, not retrieval failures.
- The tool router is now **135M**; the reader stays at **360M**. This trades 0.5 points of accuracy for a much lower latency tail.

## Latency

| Stage | Median | p95 |
|---|---:|---:|
| Middleman tool call | 1.66 s | 2.70 s |
| Retrieval | 0.12 s | 0.16 s |
| Rerank | 0.08 s | 0.11 s |
| Reader | 0.62 s | 0.75 s |
| **Total** | **2.42 s** | **3.61 s** |

Previous latency was **4.35 s** total and **20.07 s** p95. The 135M tool router plus shortcuts cut the median to **2.42 s** and the p95 to **3.61 s**. The tool-call median still mixes one-project pronoun cases that use the model with shortcut cases that do not.

## Memory and build

- Index build: **60 s**.
- GPU peak: **1.33 GB**.
- Process RSS: **2.99 GB**.
- Store: 76,145 chunks / 9,746,560 content tokens.

## Caveats

- The benchmark facts and queries are synthetic; the 200 facts are the same benchmark used throughout.
- Only 200 conversations.
- The latency was measured on a shared GPU; absolute numbers will vary.
- The tool router is 135M; the reader is 360M.
- The reader makes 5–8% errors even when retrieval is correct.
