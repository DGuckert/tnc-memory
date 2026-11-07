# Active context sizes

The large 9.75M/100M/1B numbers are **memory index sizes**, not model context. The model only sees the selected prompt.

## Model limit

- SmolLM2 context limit: **8,192 tokens**
- Tool router generation limit: **48 tokens**
- Reader generation limit: **12 tokens**

## Gateway prompts

Measured with the SmolLM2 tokenizer.

| Prompt | Median | p95 | Max |
|---|---:|---:|---:|
| 9.75M gateway tool prompt | **113** | 117 | 120 |
| Reader prompt with 5 memory notes | **175** | 182 | 186 |
| Noisy hard-conversation tool prompt | **231** | 310 | 346 |

The gateway reader sees **5 extracted memory notes plus the question**, not the full retrieved store.

## Coding/chat adapter

| Dataset | Median | p95 | Max |
|---|---:|---:|---:|
| Original chat/code rows | **132** | 354 | 974 |
| Cleaned chat/code rows | **141** | 343 | 635 |

Training used `max_len=384`; longer examples were truncated.

## Bottom line

- The retrieval index can be 9.75M, 100M, or 1B tokens.
- The model’s active context is usually **~100–350 tokens**.
- Even the hard noisy conversations stay under **350 tokens** in these tests.
