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

- The explicit-project shortcut bypasses the middleman when the user names a known project. This fixed the zero-context explicit case and removed one model call.
- Pronoun resolution is strong: **99%** correct project, **97%** correct answer.
- Typo correction is strong: **96%** correct project, **94%** correct answer.
- The remaining errors are reader mistakes, not retrieval failures.

## Latency

| Stage | Median | p95 |
|---|---:|---:|
| Middleman tool call | 2.23 s | 16.24 s |
| Retrieval | 0.25 s | 0.56 s |
| Rerank | 0.10 s | 0.13 s |
| Reader | 1.70 s | 4.24 s |
| **Total** | **4.35 s** | **20.75 s** |

The tool-call and reader times are slow on the shared 6GB GPU. The shortcut removes the tool call for explicit and typo queries. The reader is still the main cost after retrieval.

## Memory and build

- Index build: **117 s** (word + character indexes).
- GPU peak: **1.04 GB**.
- Process RSS: **2.89 GB**.
- Store: 76,145 chunks / 9,746,560 content tokens.

## Caveats

- The benchmark facts and queries are synthetic; the 200 facts are the same benchmark used throughout.
- Only 200 conversations.
- The latency was measured on a shared GPU; absolute numbers will vary.
- The middleman and reader share one 360M model.
- The reader makes 3–6% errors even when retrieval is correct.
