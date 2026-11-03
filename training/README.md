# Training set

Synthetic multi-task data for a memory-aware middleman and a memory reader.

## Files

| File | Rows | Purpose |
|---|---:|---|
| `train_sft.jsonl` | 10,000 | 5,000 tool-call + 5,000 reader examples |
| `val_sft.jsonl` | 1,500 | 750 tool-call + 750 reader examples |
| `test_sft.jsonl` | 1,500 | 750 tool-call + 750 reader examples |
| `train_reranker.jsonl` | 10,000 | query + positive + hard negatives |
| `val_reranker.jsonl` | 1,500 | held-out reranker pairs |
| `test_reranker.jsonl` | 1,500 | held-out reranker pairs |
| `manifest.json` | — | counts, schema, seeds, excluded benchmark names/codes |

## Splits

- **Train:** 1,200 project names, 10,000 SFT rows, 10,000 reranker rows.
- **Val:** 180 project names, 1,500 SFT rows, 1,500 reranker rows.
- **Test:** 180 project names, 1,500 SFT rows, 1,500 reranker rows.
- Project names and codes are disjoint across splits.
- The 200 benchmark facts used in the retrieval experiments are **excluded** (exact names and codes checked).

## Tasks

### 1. `tool_call`

Input: a multi-turn conversation. Output: JSON only.

```json
{"tool":"search_memory","args":{"project":"Groth","query":"what is the access code for project Groth?"}}
{"tool":"ask_clarification","args":{"question":"Which project do you mean?"}}
{"tool":"no_tool","args":{}}
```

The conversation may contain several projects, pronouns, “the other one,” “the new one,” or a misspelled project name. The target always uses the canonical project name.

### 2. `answer_from_memory`

Input: 5–7 memory notes plus a question. Output: the exact code, or `I don't know.` when the note is absent.

The notes include same-noun distractors (same kind of secret, different project), so the model must read the right note rather than match the noun.

### 3. Reranker pairs

```json
{"id":"rerank-train-000001","query":"what is the license key for project Fiaxgaest?","positive":"Note to self: the license key for project Fiaxgaest is S75-S9KM.","negatives":["Note to self: the license key for project Piar is WK9-J6XF.", "..."],"meta":{}}
```

Negatives are:
- 3 facts with the same noun but a different project;
- 1 fact with the same project but a different noun;
- 1 random fact.

## SFT format

Each row has:

- `id`
- `task`
- `messages` — standard chat messages; the final assistant message is the target
- `target` — the string or object the final assistant should produce
- `meta` — task metadata

Prior assistant turns are context. Train only on the final assistant message, or use a trainer that masks earlier assistant turns.

## Trained baseline

`SmolLM2-360M-Instruct` + LoRA, trained for 120 steps on 1,920 examples:

| Test, 200 each | Base | Fine-tuned |
|---|---:|---:|
| Valid tool JSON | 0.00 | 1.00 |
| Tool choice accuracy | 0.00 | 1.00 |
| Full tool-call accuracy | 0.00 | 0.97 |
| Reader accuracy | 0.21 | 0.99 |

Harder stress test, 100 conversations with real chatter noise and paraphrases: fine-tuned tool JSON **1.00**, full tool accuracy **0.86**; base **0.00**. Details: `results/training/middleman_report.md`.

The cross-encoder reranker is also trained. At 9.75M tokens it improves exact top-1 from **0.980** to **0.995** and typo top-1 from **0.430** to **0.680**. Details: `results/training/reranker_report.md`.

End-to-end retrieval + rerank + reader: exact **1.000**, typo **0.890** with entity fuzzy correction (char-only baseline: 0.595). Details: `results/training/end_to_end_report.md`.

Full gateway on 100 noisy conversations: **0.86** tool-call and **0.88** end-to-end answer accuracy. Details: `results/training/gateway_report.md`.

Full gateway on the 9.75M store: structured facts **1.000** (direct lookup); model fallback **0.950**, median **2.42 s**, p95 **3.61 s**; noisy conversations **0.94**. The tool router is a 135M LoRA; the reader is 360M. Details: `results/training/gateway_975m_report.md`.

Coding/chat pilot: 11,553 real-Discord context→next-message rows; 360M adapter reaches **24.64** held-out perplexity. Global exact-message BM25 retrieval is near zero; semantic search improves it 4–6× but is still low. Details: `results/training/chat_coding_report.md`.

Adapter: `v0.1-middleman` release. Reranker: `v0.1-reranker` release.

## Recommended use

1. Fine-tune a small model on `train_sft.jsonl` and evaluate on `val_sft.jsonl`.
2. Fine-tune a cross-encoder on the reranker files; convert listwise rows to pointwise pairs with labels 1/0.
3. Test on `test_*` only after choosing hyperparameters.
4. Mix in general chat data so the model does not become a pure JSON tool caller.

## Caveats

- The data is synthetic and template-based. It tests the format and reasoning pattern, not real-world language diversity.
- Codes and project names are random; do not treat them as real secrets.
- Real conversations, typos, aliases, and multi-turn references still need to be collected and added.
- The tool-call target assumes the gateway exposes `search_memory`, `ask_clarification`, and `no_tool`.
