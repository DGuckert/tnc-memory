# Coding and chat track

First pass at moving beyond synthetic facts into real Discord coding/chat.

## Data

- Source: 550,357 real messages across 28 channels.
- SFT rows: context = previous 3 messages, target = next message.
- Train: **11,553** rows.
- Validation: **369** rows.
- Test: **78** rows.
- Split by channel; raw text is not committed to the repo.

Coding examples use the help channels and code/technical keywords. Chat examples use general/off-topic channels.

## Chat/code adapter

Two LoRA pilots on the same data:

| Model | Val loss | Perplexity | Base loss | Base ppl |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 3.541 | 34.51 | 4.040 | 56.84 |
| SmolLM2-360M | **3.382** | **29.43** | 3.765 | 43.17 |

The 360M adapter was trained for 120 steps, 1,920 examples, about 4 minutes. Sample outputs are coherent for simple coding/chat prompts, but can repeat or drift; this is a next-message model, not an instruction-tuned assistant. Release: `v0.1-chat-360m`.

## Retrieval on real messages

Global exact-message retrieval over 551,541 messages with BM25:

| Task | recall@1 | recall@10 | recall@50 |
|---|---:|---:|---:|
| Code answer from question | 0.000 | 0.008 | 0.018 |
| Chat answer from question | 0.000 | 0.016 | 0.024 |

This is near zero. Exact next-answer retrieval at 550K messages is not a useful QA benchmark and lexical BM25 is not enough for open-domain chat/code. The next retrieval step is semantic embeddings plus channel/time scoping, not more BM25 tuning.

## What this tells us

- The fact-memory stack does not transfer directly to open-domain coding/chat.
- A 360M next-message adapter improves held-out loss and can produce coherent short replies, but it is not an assistant yet.
- Real-message retrieval needs a stronger semantic index; exact lexical matching only works for names/codes.
- Raw Discord data is privacy-sensitive; only scripts, metrics, and sanitized samples belong in the repo.

## Next

1. Collect real question/answer pairs (not just adjacent messages) for coding/chat.
2. Build a semantic message index with channel/time scoping.
3. Train or use a larger reader and evaluate with human or judge-based scoring.
4. Keep the fact-memory and chat-memory paths separate.

## Files

- `chat_coding_manifest.json` — counts.
- `chat_360m_val_metrics.json` — held-out loss/perplexity.
- `chat_360m_base_val_metrics.json` — base comparison.
- `chat_135m_val_metrics.json`, `chat_135m_base_val_metrics.json` — 135M comparison.
- `chat_next_message_retrieval.json` — next-message retrieval negative.
- `answer_retrieval_metrics.json` — code/chat answer retrieval negative.
- `chat_samples_360m.json` — sanitized sample outputs.
