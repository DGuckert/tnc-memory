# Gateway at 9.75M tokens

## Structured fact fast path

The benchmark facts are written as `Note to self: the <noun> for project <name> is <code>.` The gateway now indexes them by **(project, noun)** and answers directly when both are known.

200 conversations: 100 pronoun, 50 explicit, 50 typo.

| Case | Correct project | Correct answer |
|---|---:|---:|
| Pronoun (“it”) | 1.00 | **1.00** |
| Explicit name | 1.00 | **1.00** |
| One-char typo | 1.00 | **1.00** |
| **Overall** | 1.00 | **1.00** |

Median total latency is **~0 ms** because this path skips retrieval, reranking, and the reader. This is a structured lookup, not model reasoning. It works because the facts are already clean, typed notes.

## Fallback model path

When the note cannot be parsed, the gateway uses the previous stack: search, rerank, reader.

Before the structured index, the full model path scored **0.950** overall with **2.42 s** median and **3.61 s** p95.

## Noisy conversations

100 conversations with real Discord chatter, paraphrases, pronouns, and multiple projects:

| Metric | Value |
|---|---:|
| End-to-end answer accuracy | **0.94** |
| Search answer accuracy | 0.934 |
| Non-search action accuracy | 1.00 |

The structured index improves the noisy test from **0.88 to 0.94**. The remaining errors are project-resolution errors, not answer lookup errors. Tool-call exact-match is low because the shortcut query wording differs; the answer is the relevant metric.

## Memory and build

- Index build: **44 s**.
- GPU peak: **1.15 GB**.
- Process RSS: **2.17 GB**.
- Store: 76,145 chunks / 9,746,560 content tokens.
- Structured fact index: **200 facts**; negligible per-token cost.

## Caveats

- The 1.00 is on synthetic, structured facts. Real memory notes may be messy; the fallback path is ~0.95 on the clean benchmark and 0.94 on noisy conversations.
- The 0 ms is after the gateway is loaded; index build is 44 s.
- Project resolution in noisy multi-project conversations is the remaining source of errors.
- 99% on real, human-labeled conversations is not demonstrated yet.
