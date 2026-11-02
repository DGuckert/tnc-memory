import argparse, json, math, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def encode(tok, r, max_len):
    prefix = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
    full = prefix + r["messages"][-1]["content"] + tok.eos_token
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    f = tok(full, add_special_tokens=False)["input_ids"]
    common = 0
    for a, b in zip(p, f):
        if a != b:
            break
        common += 1
    labels = [-100] * common + f[common:]
    if len(f) > max_len:
        f = f[-max_len:]; labels = labels[-max_len:]
    return f, labels

def evaluate(model, tok, data, max_len):
    total = 0.0; ntok = 0
    model.eval(); model.config.use_cache = True
    with torch.inference_mode():
        for r in data:
            ids, labels = encode(tok, r, max_len)
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            y = torch.tensor([labels], dtype=torch.long, device="cuda")
            out = model(input_ids=x, labels=y)
            n = int((y != -100).sum())
            total += float(out.loss) * n
            ntok += n
    loss = total / max(1, ntok)
    return {"rows": len(data), "tokens": ntok, "loss": loss, "perplexity": math.exp(min(20, loss))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/home/daniel/tnc/training_runs/chat-135m-lora")
    ap.add_argument("--val", default="/home/daniel/tnc/work/chat_coding/val_chat_coding.jsonl")
    ap.add_argument("--out", default="/home/daniel/tnc/training_runs/chat-135m-lora/metrics.json")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.adapter)
    base_name = args.model or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16).to("cuda")
    model = base if args.base else PeftModel.from_pretrained(base, args.adapter)
    data = rows(args.val)
    out = evaluate(model, tok, data, args.max_len)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
