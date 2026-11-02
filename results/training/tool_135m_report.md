# 135M tool router v0.1

A smaller tool-call model for the gateway. The 360M model stays as the reader.

- **Base:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Method:** LoRA, r=16, alpha=32, 4.88M trainable params (3.50%)
- **Training:** 5,000 tool-call examples, 300 steps, batch 8, grad accumulation 2, about 11 minutes
- **Output:** `v0.1-tool-135m` release asset

## Held-out tool test

200 tool-call examples from held-out projects:

| Metric | Value |
|---|---:|
| Valid JSON | 1.00 |
| Tool choice | 1.00 |
| Full tool accuracy | **0.975** |
| Search project accuracy | 0.970 |
| Non-search accuracy | 1.00 |

The 360M tool model was **0.970** full accuracy, so the 135M model is slightly better on this test while being much faster.

## Gateway effect at 9.75M

| Metric | 360M tool | 135M tool |
|---|---:|---:|
| Answer accuracy | 0.955 | 0.950 |
| Median total | 2.62 s | **2.42 s** |
| p95 total | 20.07 s | **3.61 s** |
| GPU peak | 1.04 GB | 1.33 GB |

The 135M router trades 0.5 points of answer accuracy for a much lower latency tail.

## Caveats

- Trained only for tool calls; it is not a reader.
- Synthetic template data.
- The gateway still uses the 360M reader.
- The 135M model is only used when the gateway cannot resolve the project directly.
