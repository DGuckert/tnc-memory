# 1B-token coding test

A **1,000,000,000 content-token** logical store was tested with coding-style records. It repeats the 9.75M corpus to 7,812,500 chunks and injects 800 coding notes once.

Each coding note contains a unique project name, a unique function name, and a code snippet.

## Coding entity test

| Query type | Recall@1 |
|---|---:|
| Exact function name | **1.000** |
| One-character typo in function name | **1.000** |
| Project-scoped coding question | **1.000** |
| Structured coding fact lookup | **1.000** |
| Build time | **3.2 s** |

The coding index uses unique symbols (`project`, `function`) and code-note markers. It does **not** build a full 1B BM25/character index.

## Description-only coding test

This is the hard case: the user asks **what to do**, but does **not** name the project or function.

| Metric | Result |
|---|---:|
| Candidates per purpose | 40 |
| Recall@1 | **0.030** |
| Recall@10 | **0.250** |
| Recall@50 | **1.000** |

With no unique symbol, 40 coding notes share the same intent, so top-1 is basically random.

## Why coding goes bad

1. **Coding queries often lack unique names.**  
   “Fix CUDA OOM” is ambiguous. The same phrase can apply to many projects, models, and setups.

2. **Code snippets share vocabulary.**  
   `model`, `load`, `cuda`, `config`, and `error` appear everywhere. BM25 cannot tell which snippet is relevant.

3. **The current chat/code model is a next-message model, not an assistant.**  
   It learned to continue Discord-style chatter, not to solve coding tasks.

4. **Real-message retrieval is weak.**  
   Global exact-answer BM25 over 551K messages is near zero. Semantic search improves it but is still low.

5. **Coding needs a different memory shape.**  
   It needs code-aware parsing, file/module/function entities, error signatures, and project scoping—not just generic text chunks.

## What works

If the coding memory is stored with **unique symbols**, the 1B entity path is excellent:

- exact function/project: **100%**
- typo-corrected function: **100%**
- structured coding fact: **100%**

## What needs to change

For real coding help:

1. Parse code into entities: file, module, function, class, error signature.
2. Scope memory by project/channel/time.
3. Use semantic embeddings for description-only queries.
4. Keep coding and chat memories separate.
5. Use a larger instruction-tuned reader, not a Discord next-message model.
