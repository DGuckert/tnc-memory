import argparse, json, os, random, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_rows(path, limit=0):
    rows = []
    for line in open(path, encoding="utf-8"):
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows

class PairDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            q = r["query"]
            self.items.append((tok(q, r["positive"], truncation=True, max_length=max_len, padding=False), 1.0))
            for neg in r["negatives"]:
                self.items.append((tok(q, neg, truncation=True, max_length=max_len, padding=False), 0.0))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

def collate(batch, pad_id):
    maxlen = max(len(x["input_ids"]) for x, _ in batch)
    ids, mask, labels = [], [], []
    for x, y in batch:
        n = maxlen - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * n)
        mask.append(x["attention_mask"] + [0] * n)
        labels.append(y)
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32))

def evaluate(model, tok, path, max_len, limit=0):
    rows = load_rows(path, limit=limit)
    model.eval()
    top1 = 0; mrr = 0.0; margins = []
    with torch.inference_mode():
        for r in rows:
            passages = [r["positive"]] + r["negatives"]
            enc = tok([r["query"]] * len(passages), passages, truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            scores = model(**enc).logits.view(-1).float().cpu()
            order = torch.argsort(scores, descending=True)
            rank = int((order == 0).nonzero()[0]) + 1
            top1 += int(rank == 1)
            mrr += 1.0 / rank
            margins.append(float(scores[0] - scores[1:].max()))
    n = max(1, len(rows))
    return {"n": len(rows), "top1": top1 / n, "mrr": mrr / n,
            "mean_margin": sum(margins) / n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/home/daniel/tnc-repo/training/train_reranker.jsonl")
    ap.add_argument("--val", default="/home/daniel/tnc-repo/training/val_reranker.jsonl")
    ap.add_argument("--out", default="/home/daniel/tnc/training_runs/reranker-mini")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--val_limit", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = load_rows(args.train, args.limit)
    random.shuffle(rows)
    print("rows", len(rows), "pairs", sum(1 + len(r["negatives"]) for r in rows), flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    model.to("cuda")
    base = evaluate(model, tok, args.val, args.max_len, args.val_limit)
    print("base", base, flush=True)
    ds = PairDataset(rows, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0 = time.time(); step = 0; log = []
    for epoch in range(args.epochs):
        for ids, mask, labels in dl:
            ids = ids.to("cuda"); mask = mask.to("cuda"); labels = labels.to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=mask).logits.view(-1)
                loss = lossf(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f} elapsed {time.time()-t0:.0f}s", flush=True)
            log.append({"step": step, "loss": float(loss.item())})
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    metrics = evaluate(model, tok, args.val, args.max_len, args.val_limit)
    metrics["base"] = base
    metrics["train_rows"] = len(rows)
    metrics["pairs"] = len(ds)
    metrics["steps"] = step
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    json.dump(log, open(os.path.join(args.out, "train_log.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()
