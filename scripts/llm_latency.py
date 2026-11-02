import json, os, sys, time
import torch
sys.path.insert(0, "/home/daniel/tnc")
import llm_history_test_fast as L
from transformers import AutoTokenizer, AutoModelForCausalLM

WORK = "/home/daniel/tnc/work"
MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
DEVICE = "cuda"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model.to(DEVICE)
    model.eval()
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    facts = L.load_facts()
    filler = L.load_filler(tokenizer)
    out = {"model": MODEL, "max_ctx": max_ctx, "results": []}
    for target, n_fact_needles, n_questions in L.SIZES:
        for mode in ["plain", "tags", "index"]:
            history, fact_tok, hist_tok = L.build_history(
                tokenizer, facts, n_fact_needles, mode, target, filler, max_ctx
            )
            prompt = L.make_prompt(tokenizer, history, facts[0]["query"])
            enc = tokenizer(prompt, return_tensors="pt")
            ids = enc["input_ids"].to(DEVICE)
            attn = enc["attention_mask"].to(DEVICE)
            plen = int(ids.shape[1])
            with torch.no_grad():
                model(input_ids=ids, attention_mask=attn, logits_to_keep=1)
            pre = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=attn, logits_to_keep=1)
                torch.cuda.synchronize()
                pre.append(time.perf_counter() - t0)
            ttft = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=1,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                ttft.append(time.perf_counter() - t0)
            row = {
                "target_tokens": target, "mode": mode,
                "n_fact_needles": n_fact_needles, "n_questions": n_questions,
                "history_tokens": hist_tok, "fact_tokens": fact_tok, "prompt_tokens": plen,
                "prefill_ms_median": sorted(pre)[len(pre) // 2] * 1000,
                "prefill_ms_min": min(pre) * 1000,
                "ttft_ms_median": sorted(ttft)[len(ttft) // 2] * 1000,
                "prefill_tokens_per_s": plen / sorted(pre)[len(pre) // 2],
                "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            out["results"].append(row)
            print(json.dumps(row), flush=True)
            with open(os.path.join(WORK, "llm_latency.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1)
    print("done")

if __name__ == "__main__":
    main()
