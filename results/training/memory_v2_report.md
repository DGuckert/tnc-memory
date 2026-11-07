# TNC memory v2: return notes, not answers

This round changes the shape of the system and replaces several synthetic results with real-data ones.

```
conversation
  -> EntityTracker (mention state, focus, ordinals, negation, typo correction, ask-when-ambiguous)
  -> scoped retrieval
       facts:    typed (entity, slot) lookup + BM25/char + cross-encoder rerank
       messages: BM25 + bge-small (message and context window) -> RRF -> cross-encoder rerank
       code:     symbol/file/error/dependency/command records -> BM25 on split identifiers + bge -> RRF
  -> memory_api.retrieve() returns {entity, resolution, needs_clarification, structured, notes, context}
  -> the HOST model reads the notes (any size); the 360M reader is out of the critical path
```

## Headline numbers

| What | Before | Now | Notes |
|---|---:|---:|---|
| Noisy multi-project conversations, end-to-end (100) | 0.94, 6 wrong answers | **0.97, 0 wrong answers** | 3 rows now ask "which project?" instead of guessing |
| Reference resolution, old hard set (93) | 0.828 | **0.968** | independent set written before this work |
| Reference resolution, held-out phrasings (400) | 0.565 | **0.978** | frozen tracker, evaluated once |
| Real Discord recall, 205 questions over 603K messages, exact r@1 | "near zero" (different task) | **0.951** (r@10 0.980) | lexically easy set, see caveats |
| Same, low-overlap questions only (22) | | **0.86** r@1 | BM25 alone 0.23 |
| Vague coding queries, 79 q over 14.3K code records, r@1 / r@10 | 0.030 (synthetic, chance by construction) | **0.73 / 0.96** repo-scoped | hybrid, no rerank |
| Notes for the 360M reader | 5 | **5 is optimal** | it copies note 1, shuffled k=50 gives 0.27 |
| gemma-3-1b as host reader | | 0.91 at k=1, 0.33 at k=20 | too small; a 7B+ host is still untested |
| Trained routers on held-out references | | 0.300 (135M) / 0.275 (360M) | the tracker gets 0.978 |

## 1. The top_k sweep

Full write-up: `topk_sweep_report.md`. With the structured shortcut off, the 360M reader scores 0.92 at k=1, **0.97 at k=5**, 0.96 at k=10/20 and 0.945 at k=50. Shuffle the same notes and it falls to 0.81 (k=10), 0.58 (k=20) and 0.27 (k=50). It is position-copying, not reading. More context only helps a reader that actually reads, so the middleman now returns notes and a host model reads them.

## 2. Reference resolution: `scripts/entity_tracker.py`

The old gateway resolved projects with three regex shortcuts. Given "the other one / new one / first one", it returned nothing unless only one project had been mentioned. The tracker keeps explicit state:

- **mentions**: every known entity, with turn and polarity. "not Grost" is recorded but never used as a referent.
- **intro order** for "first / original / second / new / latest".
- **focus**: the entity last *talked about*. Earlier reference questions are resolved in sequence, so "the other one?" then "and its key?" follows the chain.
- **"the other one"** is unique only when exactly one non-focus entity exists. Otherwise it returns `clarify` with the candidates.
- **typos**: 1-2 edit OSA against known entities, never on words from a general vocabulary (corpus words seen 20+ times). Without that list, "This" got corrected into the project "Thias".
- **slot-aware focus**: with known attribute names, only turns that ask for one move the focus. Chatter that says "it" or "the other" no longer hijacks it.

| Set | n | old heuristics | tracker | 135M router | 360M middleman |
|---|---:|---:|---:|---:|---:|
| hard (independent) | 93 | 0.828 | **0.968** | 0.828 | 0.860 |
| reference (tuned on) | 400 | 0.527 | 1.000 | 0.365 | 0.352 |
| **held-out phrasings (frozen)** | 400 | 0.565 | **0.978** | 0.300 | 0.275 |

Tracker latency is under 1 ms per query (CPU). The trained routers (135M tool router, 360M middleman, both LoRA, 48-token JSON generation) do worse than the old regex on the new sets. They never output `ask_clarification` there, and they mostly fail "other / first / second / negation" references. They were trained on the earlier synthetic distribution, and these phrasings are out of distribution for them. Model-based routing should come back only if it beats the tracker on real conversations.

Honest notes:
- I wrote both `make_reference_eval.py` and the tracker, then iterated against the reference set, so its 1.000 is a development score. The held-out set (new templates, new seed, written after the tracker was frozen) is the real number: **0.978**. 7 of its 9 misses are one phrasing the tracker doesn't know ("I don't mean X, the other one"). One is a typo halfway between two real names, and one is a project named like an English word ("Castes"). After the run I added that negation phrasing (plus "other than" and "besides"), and held-out rose to 0.995. That number is post-hoc, not blind (`resolve_posthoc.json`). The frozen result is kept in `resolve_metrics_frozen.json`.
- The 3 hard-set misses are "the other one" with **three** projects mentioned. The set's label is the earliest project. The tracker asks instead, which I think is the right behaviour, but it is scored as wrong.

## 3. Noisy conversations end-to-end: `scripts/hard_e2e_v2.py`

Same 100 conversations and the same scoring as `hard_gateway_eval.py`, run through `memory_api.retrieve()`:

| | v1 gateway | v2 (`retrieve`) |
|---|---:|---:|
| end-to-end accuracy | 0.94 | **0.97** |
| wrong answers | 6 | **0** |
| asked for clarification instead | 0 | 3 |
| latency | 2.4 s median (model path) | ~30 ms (resolve + rerank) |

Before the general-vocabulary fix it was 0.95 with 1 wrong answer ("This" -> "Thias"). Both results are saved (`hard_e2e_v2_before_vocabfix.json`).

## 4. Real-message retrieval: `scripts/message_index.py`

**The old "real-message retrieval is near zero" was measuring the wrong task.** `chat_retrieval_eval.py` used a message as the query and the *next* message as the target. That's next-message prediction, and no memory system should solve it. This round adds a recall eval: 205 questions written the way you'd ask a memory system ("the vllm error on wsl about cpu offloading needing pinned memory"), each targeting one message among **603,531 Discord messages** in 28 channels.

| Retriever (global, exact message) | r@1 | r@10 | median latency |
|---|---:|---:|---:|
| BM25 | 0.741 | 0.893 | 2 ms |
| bge-small, message | 0.790 | 0.922 | 13 ms |
| bge-small, 3-message context window | 0.102 | 0.449 | 13 ms |
| hybrid RRF (BM25 + both dense) | 0.776 | 0.951 | 30 ms |
| hybrid + rerank on context window | 0.376 | 0.976 | 87 ms |
| **hybrid + rerank on the message** | **0.951** | **0.980** | 140 ms |
| same, restricted to the right channel | 0.966 | 0.990 | |

- **Context-window passages point at the wrong message.** The window ending at message *i+1* contains message *i*, so the next message out-scores the target. Same-exchange accuracy for "rerank on window" is 0.93 even though exact accuracy is 0.38. Rerank on the message itself.
- **Scoping.** Restricting to the right channel adds 1-2 points. Author scoping on the 15 author questions: 0.93 with rerank either way. Without rerank it *lowered* r@1 from 0.73 to 0.47 (4 questions). Unexplained, small n.

**Caveat, and a big one: the questions were written while looking at the target message.** 183/205 share four or more content words with it. Split by overlap:

| Shared content words | n | BM25 r@1 | dense r@1 | hybrid+rerank r@1 |
|---|---:|---:|---:|---:|
| 0-3 | 22 | 0.23 | 0.64 | **0.86** |
| 4+ | 183 | 0.80 | 0.81 | **0.96** |

Lexical search falls apart on the low-overlap questions, and semantic + rerank carries them. The fair test is questions written **from memory, without seeing the message**. That is the human-labelled set this project still needs.

## 5. Code-aware memory: `scripts/code_index.py`

Parses repos into function / method / class records (name, signature, docstring, calls, raised and logged error strings, body head), plus file, error, dependency and command records. Python uses `ast`; Java/Kotlin/JS/TS use brace-matched regex. Indexed: your 7 repos (voice-assistant, bus-alarm, jarvis-launcher, ButlerMusic, Discord-ai-response, ChessForge, tnc-memory) **plus two large third-party clones (flowsint, Strata) as distractors**: 14,329 records, about 50 s to parse over `/mnt/c`.

79 vague questions with no function or file names ("the interceptor that logs the user out on a 401 except for the login call itself"):

| | global r@1 | global r@10 | repo-scoped r@1 | repo-scoped r@10 |
|---|---:|---:|---:|---:|
| BM25 (split identifiers) | 0.633 | 0.886 | 0.722 | 0.937 |
| bge-small | 0.595 | 0.873 | 0.658 | 0.937 |
| **hybrid RRF** | 0.684 | 0.899 | **0.734** | **0.962** |
| hybrid + rerank (ms-marco) | 0.633 | 0.924 | 0.696 | 0.937 |

- Repo scoping (from the tracker, or from the cwd) is worth about 5 points of r@1.
- The ms-marco reranker is trained on web passages. It helps r@5 but hurts r@1 on code, so the default is hybrid without rerank.
- Same caveat as above: I wrote the questions from the function bodies, so domain words leak ("gtfs", "spotify").

**The old "description-only coding recall@1 = 0.030" was not a retrieval result.** `coding_1b_no_name_eval.py` ranks 40 notes with identical wording by their position in the index, so it is chance by construction. Once the project is known (repo scope), that task is trivial.

## 6. `memory_api.retrieve()`

One call, used by every eval above:

```python
mem = Memory(fact_docs=docs, message_index=MessageIndex(), code_index=CodeIndex(REPOS))
out = mem.retrieve(messages, k=5)               # or scope={"channel": ..., "author": ..., "repo": ...}
if out["needs_clarification"]: ask(out["clarify"])
elif out["structured"]: answer = out["structured"]["value"]        # typed fact, no model
else: host_prompt = out["context"] + question                      # host model reads the notes
```

**Host-reader result (gemma-3-1b, details in `topk_sweep_report.md`):** 0.91 at k=1, falling to 0.33 at k=20. A 1B general model over-refuses and slips on copying, so it is a worse reader than the fine-tuned 360M. Prefill is not the constraint: 3.3K to 7.7K tok/s, so 100 notes take 340 ms. Whether more notes help a *capable* host is still unmeasured and needs a 7B+ model. Until then: use `structured` when present, and k=5 otherwise.

## 7. Instruction-style data

`scripts/mine_qa_pairs.py` pairs help-seeking messages with the next substantive reply from someone else: 27,291 pairs, 12,703 technical. **It isn't usable yet.** In a spot check, 4 of 5 technical pairs answered a different conversation. Busy channels interleave, and the export has no reply-to field. Re-export with Discord reply references before training anything on this. With a real host model reading notes, a fine-tuned small reader is no longer on the critical path anyway.

## What is still open

1. **Human-written recall questions.** About 100 asked from memory, without looking at the message. That is the one result that would settle the real-message number.
2. **Entity names that are also English words** ("Castes"), and typos equidistant from two names. Both should ask for clarification instead of picking one.
3. **A larger host model.** Qwen3-30B-A3B or Qwen3.6-35B at k = 5/20/50 on a subset. Not run here because it needs ~17-20 GB of RAM while the desktop was in use.
4. **Write-time structuring** of free-form messages into (entity, slot, value) facts with a local LLM, so the structured path works beyond the synthetic template.
5. **A code reranker** (or no reranker) for code. ms-marco hurts r@1 there.
