# End-to-end memory pipeline

Full stack at 9.75M tokens:

```
question -> word BM25 / character n-gram -> top 50
         -> fine-tuned cross-encoder -> top 5 notes
         -> middleman LoRA reader -> answer
```

## Results

200 exact-name queries and 200 one-character-typo queries:

| Query path | Gold in top 5 | Reader correct when gold present | End-to-end correct |
|---|---:|---:|---:|
| Exact name | **1.000** | **1.000** | **1.000** |
| Typo, char search | 0.755 | 0.788 | **0.595** |
| Typo, char + canonical query | 0.755 | 0.894 | **0.675** |
| Typo, entity fuzzy + canonical rerank | **0.905** | **0.983** | **0.890** |

- Exact-name questions are solved end-to-end: **200/200** correct.
- The typo path was the weak point. Character search plus canonicalization reached **67.5%**.
- Matching the misspelled token against the stored project-name list (Damerau-Levenshtein, distance ≤2) and then reranking with the canonical query raises typo end-to-end accuracy to **89.0%**.
- The entity list is only **810 project names**; this is cheap and does not change the memory budget.
- The remaining typo loss is mostly entity correction: 94.0% of gold facts are in the top 50 before reranking, and the corrected top-5 contains the gold **90.5%** of the time.

## What this proves

- The retrieval, reranking, and reader components work together.
- The exact-name path is production-shaped.
- The typo path is the remaining weak point: **0.595** end-to-end.
- A middleman model is not needed for explicit names; it is needed for pronouns and conversation references.

## Caveats

- The benchmark is synthetic and template-based.
- The reader is a 360M LoRA; a larger reader may handle noisy notes better.
- Only 200 queries per condition.
- The reader sees five extracted fact notes, not the full raw chunks.

## Files

- `end_to_end_975m.json` — char-path metrics.
- `entity_typo_975m.json` — entity-fuzzy typo metrics.
- `scripts/end_to_end_eval.py` — char-path evaluation.
- `scripts/entity_end_to_end.py` — entity-fuzzy typo evaluation.
