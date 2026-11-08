# Memory that behaves like context: the life-log eval

> **Status: in progress.** Small-log results for `cards_oracle`, `cards` and `rag` are final. `full` (true full context) and `agent` are still running on the small log, and then the 600K-token log runs. This file is updated as they land.

## Why this eval exists

The v2 numbers (0.99 on the 9.75M store, 0.95 r@1 on real messages) are all **single-fact lookups**. Single-fact lookup is exactly what RAG is good at, so those numbers can't tell a memory layer from plain RAG. Things that only context could do were never tested:

- facts that **change** (which value is current?),
- questions over **many facts at once** ("which projects use Postgres?", "how many are paused?"),
- **chains** ("what language is the project on port 5035?"),
- **time** ("which did I start first?").

## The test world (`scripts/life_log.py`)

- **16 invented projects over 10 months** (2026-01-05 to 2026-10-30).
  - Each has a language, database, host, port, status and collaborators.
  - Ports move, databases migrate, hosts change, projects pause, resume, ship or die, and people join and leave.
  - 146 events in total.
- **Messages are casual and varied**, e.g. "migrated Venmire off Postgres onto MariaDB". About 40% of some event types are split into a lead message plus a follow-up that only says "it", e.g. "moved it from :7679 to :7764".
- **The events are buried in real Discord chatter** (filler stays local, never committed):
  - **small log:** 314 messages, about 7K tokens, so it fits in a context window and true full context can be measured;
  - **big log:** 30,164 messages, about 600K tokens, so only a memory layer can cope.
- **Same seed for both logs**, so the projects, events and questions are identical. The gold answers come from the event log, never from text.
- **Seven question types.** 12 questions are generated per type and 8 per type are run, so 56 questions.

| type | example |
|---|---|
| current value | What port is Drisk on right now? |
| past value | Before Quillon's host was basalt, what was it? |
| full history | List every host Ferrow has had, oldest first. |
| which projects… | Which projects use Postgres right now? |
| how many… | How many of my projects are paused right now? |
| multi-hop | What language is the project currently on port 5035 written in? |
| which first… | Which did I start first, Umbrin or Zephram? |

Scoring is strict:
- **sets** must match exactly, with no extras;
- **histories** must match in the right order;
- **single values** count the first value of the right kind named in the final answer;
- **counts** take the first number.

## What the memory layer does differently from RAG (`scripts/lifelog_memory.py`)

1. **Write-time consolidation.**
   - Every message is read once, as it arrives, by the host LLM.
   - The LLM writes typed facts in a grammar-constrained one-line-per-message format, e.g. `Drisk port=7764`.
   - A fact store keeps each `(project, attribute)` with its full dated history. Later facts supersede earlier ones, and collaborators are tracked as joins and leaves.
2. **Cards.**
   - One compact line per project: current values, what they were before, and when they changed.
   - All 16 cards come to about 1.5K tokens. Their size grows with the number of entities, not the amount of chatter: the 600K-token log gives the same size of cards.
3. **Aggregate queries.** `find_projects(attr, value, as_of)` scans the whole store instead of taking a top-k.
4. **Agent mode.**
   - The host calls `search_messages`, `get_project`, `find_projects` and `list_projects` in a loop (up to 6 steps).
   - This lets it chain lookups, e.g. find the project on a port, then look up its language.

**Extraction on the big log.** A cheap keyword/entity gate sends only 2,290 of 30,164 messages (7.6%) to the extractor, plus the message after any gated one when it says "it". The gate's keyword list is generic project vocabulary, but it was written knowing the domain. Say so if you quote this.

## Results so far (small log, Qwen3-30B-A3B-Thinking Q4_K_M host)

![by type](lifelog_by_type_small.png)

| condition | overall | current | past | history | which… | how many | multi-hop | which first | prompt tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG (top-10 messages) | **0.43** | 0.75 | 0.63 | 0.38 | 0.13 | 0.25 | 0.13 | 0.75 | 396 |
| cards (extracted) | **0.70** | 0.88 | 1.00 | 0.88 | 0.38 | 0.38 | 0.50 | 0.88 | 3,158 |
| cards (perfect extraction) | **0.77** | 1.00 | 1.00 | 1.00 | 0.50 | 0.75 | 0.38 | 0.75 | 2,862 |
| full context | *running* | | | | | | | | |
| agent | *running* | | | | | | | | |

**Extraction quality** (small log, every message sent to the extractor, 32 calls, 4.4 min):
- 87.5% of current values are exactly right;
- 83% of attribute histories are exactly right;
- 1 junk entity was created;
- 4 facts were dropped because the project name appeared nowhere in what the extractor saw.

![overall](lifelog_overall.png)

## What it means so far

- **Consolidated memory beats RAG by 27 points**, 0.70 vs 0.43. The gap is biggest exactly where RAG can't work:
  - history: 0.88 vs 0.38;
  - multi-hop: 0.50 vs 0.13;
  - "which projects": 0.38 vs 0.13.

  RAG returns a slice of messages, often stale ones. For "which projects use Postgres?" it returned three "uses Postgres" messages and a migration note, with no way to tell which projects still use it.
- **Extraction costs about 7 points**, perfect cards 0.77 vs extracted 0.70. Most of that loss is on the counting and "which" questions, where one missed status change flips the answer.
- **Even perfect cards get only 0.50 on "which projects" and 0.38 on multi-hop.** Every fact needed is in the prompt, so these are reasoning failures, not memory failures: the model has to scan 16 cards and combine facts in its head. Typical misses:
  - listing 3 of 4 projects;
  - answering "I don't know" when the card is right there;
  - counting 8 when the answer is 11.

  This is the motivation for agent mode, where `find_projects` does the scan exactly.
- **Still not true context.** The store only consolidates the 7 attributes it knows about. "Why did I drop Postgres?" falls back to message search.

## Caveats

- **Synthetic events.** The event messages are templated (varied, casual, with pronoun follow-ups); the filler is real.
- **The host is a reasoning checkpoint used without its think block.**
  - Qwen3-Thinking-2507 reasons out loud anyway, so every condition lets it reason and reads its final `ANSWER:` line, with a forced continuation if it runs out of tokens.
  - Every condition gets the same treatment.
- **About 25–30 s per question on this PC** (RAM-starved: the experts thrash from disk on every prefill). That cost is the hardware, not the method.
- **8 questions per type**, so one question is 12.5 points within a type. Read the per-type numbers as directional.
