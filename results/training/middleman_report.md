# Middleman LoRA v0.1

Fine-tuned `HuggingFaceTB/SmolLM2-360M-Instruct` on the synthetic middleman/reader training set.

- **Adapter:** `v0.1-middleman` release asset
- **Base model:** `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Method:** LoRA, r=16, alpha=32, dropout=0.05, 8.68M trainable params (2.34%)
- **Training:** 120 steps, batch 4, grad accumulation 4, max length 256, lr 2e-4
- **Effective examples seen:** 1,920 of 10,000 (about 0.2 epoch)
- **Time:** ~21 minutes on the RTX 2060 6GB

## Results

Validation, 100 tool + 100 reader:

| Metric | Value |
|---|---:|
| Valid JSON | 1.00 |
| Tool choice accuracy | 1.00 |
| Full tool-call accuracy | 0.96 |
| Reader accuracy | 0.99 |

Test, 200 tool + 200 reader, held-out projects:

| Metric | Value |
|---|---:|
| Valid JSON | 1.00 |
| Tool choice accuracy | 1.00 |
| Full tool-call accuracy | 0.97 |
| Search: project accuracy | 0.964 |
| Search: full query accuracy | 0.964 |
| Non-search: full accuracy | 1.00 |
| Reader accuracy | 0.99 |

Base model, 100 tool + 100 reader:

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Valid JSON | 0.00 | 1.00 |
| Tool choice accuracy | 0.00 | 1.00 |
| Full tool-call accuracy | 0.00 | 0.97 |
| Reader accuracy | 0.21 | 0.99 |

## What it learned

- Emit strict JSON tool calls.
- Resolve pronouns and “the other one” to the correct project.
- Ask for clarification when no project is available.
- Return `no_tool` for small talk.
- Read a note and return the exact code.
- Say `I don't know.` when the note is absent.

## Caveats

- The data is synthetic and template-based. Test and train share templates, so these numbers are an upper bound on real-world performance.
- The model saw only ~0.2 epoch; more data and more steps may improve robustness, but the loss already dropped to near zero, so overfitting is possible.
- Only 200 held-out examples per task were evaluated.
- The model is a 360M reader; a larger reader or the host model may be better for hard language.
- The adapter is a LoRA, not a full model. Load it on the base model.

## Files

- `val_metrics.json` — validation metrics.
- `test_metrics.json` — held-out test metrics.
- `base_test_metrics.json` — base-model comparison.
- `train_log.json` — step/loss log.
- `adapter_config.json` — LoRA configuration.
- Release: https://github.com/DGuckert/tnc-memory/releases/tag/v0.1-middleman
