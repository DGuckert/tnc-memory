# 1B-token scale test

A **1,000,000,000 content-token** store was tested as a logical scale check. It repeats the 9.75M corpus to 7,812,500 chunks and injects 800 facts once. This is an **entity/structured-fact scale test**, not a full BM25 + character-index run.

## Entity index

| Metric | Value |
|---|---:|
| Store size | 1,000,000,000 content tokens |
| Chunks | 7,812,500 |
| Unique entities indexed | 800 |
| Build time | **2.8 s** |
| Exact-name recall@1 | **1.000** |
| Typo + entity correction recall@1 | **0.935** |
| Structured fact lookup accuracy | **1.000** |

The entity index scans the 1B logical store for `Note to self` records and indexes the project name. It does not build the full 1B word/character indexes.

## Structured gateway

200 conversations: 100 pronoun, 50 explicit, 50 typo.

| Case | Accuracy |
|---|---:|
| Pronoun (“it”) | **1.000** |
| Explicit name | **1.000** |
| One-char typo | **1.000** |
| **Overall** | **1.000** |

The structured gateway resolves the project and reads the fact directly; it does not need the full retrieval stack.

## What this means

- The structured-fact path scales to 1B logical tokens.
- Exact-name retrieval is effectively solved by an entity index.
- Typo correction is strong at 1B (**93.5%**) even without the full character fallback.
- The full 1B word + character index was **not** run; the character index alone would be ~36.8GB at the current estimate.
- This is still a repeated-filler scale test, not 1B unique conversations.

## Next

For real 1B unique content, use:
1. entity/structured-fact indexing,
2. semantic retrieval with channel/time scoping,
3. deduplicated storage for repeated content.
