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
| Tool-call full accuracy | 0.86 |
| End-to-end answer accuracy | **0.88** |
| Search answer accuracy | 0.868 |
| Non-search action accuracy | 1.00 |

End-to-end is slightly higher than tool-call full accuracy because some tool calls differ only in query wording while still returning the correct code. The score that matters is the answer.

## 9.75M store

200 conversations against the full 9.75M-token store:

| Case | Correct answer |
|---|---:|
| Pronoun (“it”) | **0.97** |
| Explicit name | **0.94** |
| One-char typo | **0.94** |
| **Overall** | **0.955** |

Median total latency is **2.62 s** after shortcut and generation optimizations; GPU peak **1.04 GB**, RSS **2.89 GB**. Details: `gateway_975m_report.md`.

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
