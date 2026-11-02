import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

def load_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]

def generate(model, tok, row, max_new_tokens=64):
    prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def evaluate(model, tok, path, limit):
    rows = load_rows(path)
    tool = [r for r in rows if r["task"] == "tool_call"][:limit]
    reader = [r for r in rows if r["task"] == "answer_from_memory"][:limit]
    model.eval()
    model.config.use_cache = True
    t_json = t_tool = t_full = 0
    s_n = s_proj = s_full = 0
    o_n = o_full = 0
    t_ex = []
    for r in tool:
        text = generate(model, tok, r)
        target = r["target"]
        pred = None
        try:
            pred = json.loads(text)
            t_json += 1
        except Exception:
            pass
        if isinstance(pred, dict) and pred.get("tool") == target.get("tool"):
            t_tool += 1
            if target.get("tool") == "search_memory":
                s_n += 1
                pa = pred.get("args", {}); ta = target.get("args", {})
                if pa.get("project") == ta.get("project"):
                    s_proj += 1
                if pa.get("project") == ta.get("project") and pa.get("query") == ta.get("query"):
                    s_full += 1
                    t_full += 1
            else:
                o_n += 1
                o_full += 1
                t_full += 1
        if len(t_ex) < 8:
            t_ex.append({"target": target, "pred": text})
    r_ok = 0
    r_ex = []
    for r in reader:
        text = generate(model, tok, r)
        if text.strip() == r["target"].strip():
            r_ok += 1
        if len(r_ex) < 8:
            r_ex.append({"target": r["target"], "pred": text})
    n_t = max(1, len(tool)); n_r = max(1, len(reader))
    return {
        "tool_n": len(tool), "tool_valid_json": t_json / n_t,
        "tool_tool_acc": t_tool / n_t, "tool_full_acc": t_full / n_t,
        "search_n": s_n, "search_project_acc": s_proj / max(1, s_n),
        "search_full_acc": s_full / max(1, s_n),
        "other_n": o_n, "other_full_acc": o_full / max(1, o_n),
        "reader_n": len(reader), "reader_acc": r_ok / n_r,
        "tool_examples": t_ex, "reader_examples": r_ex,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/home/daniel/tnc/training_runs/middleman-lora")
    ap.add_argument("--test", default="/home/daniel/tnc-repo/training/test_sft.jsonl")
    ap.add_argument("--out", default="/home/daniel/tnc/training_runs/middleman-lora/test_metrics.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--base", action="store_true")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    base.to("cuda")
    if args.base:
        model = base
    else:
        model = PeftModel.from_pretrained(base, args.adapter)
    metrics = evaluate(model, tok, args.test, args.limit)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("examples")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
