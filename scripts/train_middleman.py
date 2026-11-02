import argparse, json, math, os, random, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

BASE = "/home/daniel/tnc"
MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

def load_rows(path, task=None, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if task and r["task"] != task:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows

class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for r in rows:
            msgs = r["messages"]
            prefix = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = prefix + msgs[-1]["content"] + tokenizer.eos_token
            p = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            f = tokenizer(full, add_special_tokens=False)["input_ids"]
            common = 0
            for a, b in zip(p, f):
                if a != b:
                    break
                common += 1
            labels = [-100] * common + f[common:]
            if len(f) > max_len:
                f = f[-max_len:]
                labels = labels[-max_len:]
            self.items.append((f, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    ids, labels, mask = [], [], []
    for x, y in batch:
        n = maxlen - len(x)
        ids.append(x + [pad_id] * n)
        labels.append(y + [-100] * n)
        mask.append([1] * len(x) + [0] * n)
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))

def make_model(model_name=MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer

def evaluate(model, tokenizer, path, limit, max_new_tokens=64):
    rows = load_rows(path)
    tool = [r for r in rows if r["task"] == "tool_call"][:limit]
    reader = [r for r in rows if r["task"] == "answer_from_memory"][:limit]
    model.eval()
    model.config.use_cache = True

    def generate(r):
        prompt = tokenizer.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return text

    tool_ok = 0; tool_json = 0; tool_project = 0; tool_full = 0
    tool_examples = []
    for r in tool:
        text = generate(r)
        target = r["target"]
        try:
            pred = json.loads(text)
            tool_json += 1
        except Exception:
            pred = None
        if isinstance(pred, dict) and pred.get("tool") == target.get("tool"):
            tool_ok += 1
            if pred.get("tool") == "search_memory":
                p_args = pred.get("args", {})
                t_args = target.get("args", {})
                if p_args.get("project") == t_args.get("project"):
                    tool_project += 1
                if p_args.get("project") == t_args.get("project") and p_args.get("query") == t_args.get("query"):
                    tool_full += 1
            else:
                tool_full += 1
        if len(tool_examples) < 5:
            tool_examples.append({"target": target, "pred": text})

    reader_ok = 0
    reader_examples = []
    for r in reader:
        text = generate(r)
        if text.strip() == r["target"].strip():
            reader_ok += 1
        if len(reader_examples) < 5:
            reader_examples.append({"target": r["target"], "pred": text})

    n_t = max(1, len(tool)); n_r = max(1, len(reader))
    metrics = {
        "tool_n": len(tool), "tool_valid_json": tool_json / n_t,
        "tool_tool_acc": tool_ok / n_t, "tool_project_acc": tool_project / n_t,
        "tool_full_acc": tool_full / n_t,
        "reader_n": len(reader), "reader_acc": reader_ok / n_r,
        "tool_examples": tool_examples, "reader_examples": reader_examples,
    }
    model.config.use_cache = False
    model.train()
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/home/daniel/tnc-repo/training/train_sft.jsonl")
    ap.add_argument("--val", default="/home/daniel/tnc-repo/training/val_sft.jsonl")
    ap.add_argument("--out", default="/home/daniel/tnc/training_runs/middleman-lora")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--task", default="")
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval_limit", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)

    rows = load_rows(args.train, task=args.task or None, limit=args.limit or None)
    random.shuffle(rows)
    print("train rows", len(rows), "model", args.model, "task", args.task or "all", flush=True)
    model, tokenizer = make_model(args.model)
    ds = SFTDataset(rows, tokenizer, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.eos_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    step = 0; micro = 0; t0 = time.time()
    log = []
    while step < args.max_steps:
        for ids, mask, labels in dl:
            ids = ids.to("cuda"); mask = mask.to("cuda"); labels = labels.to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
                step += 1
                if step % 25 == 0:
                    print(f"step {step} loss {loss.item()*args.grad_accum:.4f} "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)
                log.append({"step": step, "loss": float(loss.item() * args.grad_accum)})
                if step % args.save_every == 0 or step == args.max_steps:
                    model.save_pretrained(args.out)
                    tokenizer.save_pretrained(args.out)
                    json.dump(log, open(os.path.join(args.out, "train_log.json"), "w"), indent=1)
                    print("saved", args.out, flush=True)
                if step >= args.max_steps:
                    break
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    metrics = evaluate(model, tokenizer, args.val, args.eval_limit)
    metrics["train_rows"] = len(rows)
    metrics["steps"] = step
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("examples")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
