# Memory gateway

`gateway.py` is the first full middleman service:

```
conversation
  -> middleman LoRA (tool call: search / clarify / no_tool)
  -> fuzzy entity correction, then word BM25 or character n-gram search
  -> canonicalize query from the top candidate
  -> cross-encoder reranker (top 5)
  -> middleman LoRA reader
  -> answer + trace
```

## Hard conversation test

100 conversations with real Discord chatter as noise, paraphrased questions, typos, pronouns, and “the other one” references. The memory store is the 80 hard-eval projects (one fact each).

| Metric | Value |
|---|---:|
| Tool-call exact accuracy | 0.16 |
| End-to-end answer accuracy | **0.94** |
| Search answer accuracy | 0.934 |
| Non-search action accuracy | 1.00 |

Tool-call exact accuracy is low because the shortcut query wording differs from the target; the answer is the relevant metric. The structured fact index raised noisy-conversation answer accuracy from **0.88 to 0.94**.

## 9.75M store

200 conversations against the full 9.75M-token store:

| Case | Correct answer |
|---|---:|
| Pronoun ("it") | **1.00** |
| Explicit name | **1.00** |
| One-char typo | **1.00** |
| **Overall** | **1.000** |

This is the structured-fact fast path: the gateway parses the `Note to self` facts and indexes them by project and noun. It skips retrieval and reader, so latency is ~0 ms. This is a lookup, not model reasoning. The model fallback was **0.950** with **2.42 s** median / **3.61 s** p95. Details: `gateway_975m_report.md`.

## What it demonstrates

- The middleman resolves pronouns and references from real-looking conversation noise.
- The gateway retrieves from the memory store.
- The reranker selects the note.
- The reader returns the exact code.
- Clarification and small-talk actions work.

## Caveats

- The hard-eval store is small (80 facts), so retrieval is easier than 9.75M.
- The labels are synthetic; only the filler chatter is real.
- The gateway loads both the 360M middleman and the 22M reranker on one GPU.
- The 9.75M end-to-end result is in `end_to_end_report.md`: exact **1.000**, typo **0.675**.

## Files

- `scripts/gateway.py` — reusable gateway.
- `scripts/hard_gateway_eval.py` — hard conversation test.
- `hard_gateway_metrics.json` — metrics.
