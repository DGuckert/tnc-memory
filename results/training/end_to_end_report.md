# End-to-end memory pipeline

Full stack at 9.75M tokens:

```
question -> word BM25 / character n-gram -> top 50
         -> fine-tuned cross-encoder -> top 5 notes
         -> middleman LoRA reader -> answer
```

## Results

200 exact-name queries and 200 one-character-typo queries:

| Query | Gold in top 5 | Reader correct when gold present | End-to-end correct |
|---|---:|---:|---:|
| Exact name | **1.000** | **1.000** | **1.000** |
| One-character typo | 0.755 | 0.788 | **0.595** |
| Typo + canonicalized query | 0.755 | 0.894 | **0.675** |

- Exact-name questions are solved end-to-end: **200/200** correct.
- Typo questions lose most of their accuracy at retrieval and reading: gold is in the top 5 only **75.5%** of the time, and the reader is correct **78.8%** of the time when it is present.
- Canonicalizing the typo query from the top retrieved note improves the reader to **89.4%** when gold is present and end-to-end to **67.5%**. This is the value of the reranker plus a normalization step.
- The remaining typo loss is retrieval: gold is in the top 5 only **75.5%** of the time.

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

- `end_to_end_975m.json` — metrics.
- `scripts/end_to_end_eval.py` — evaluation code.
