import json, os, re, math, random, time
from collections import defaultdict, Counter
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work", "chat_coding")
MODEL = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
K = 50
BANK = 20000

def tokenize(s):
    return re.findall(r"[a-z0-9_]+", s.lower())

def main():
    channels = defaultdict(list)
    for line in open(os.path.join(BASE, "corpus.jsonl"), encoding="utf-8"):
        o = json.loads(line)
        t = re.sub(r"\s+", " ", o["t"]).strip()
        if len(t) >= 20:
            channels[o["ch"]].append(t)
    all_msgs = [t for v in channels.values() for t in v]
    print("all", len(all_msgs), flush=True)
    device = "cuda"
    model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 256
    out = {}
    for kind in ("code", "chat"):
        rows = [json.loads(line) for line in open(os.path.join(WORK, f"answer_{kind}.jsonl"), encoding="utf-8")]
        targets = list(dict.fromkeys(r["target"] for r in rows))
        rng = random.Random(9000 + (0 if kind == "code" else 1))
        pool = targets[:]
        rng.shuffle(pool)
        seen = set(targets)
        while len(pool) < BANK:
            t = rng.choice(all_msgs)
            if t not in seen:
                seen.add(t); pool.append(t)
        bank = pool[:BANK]
        # guarantee every target is present
        for t in targets:
            if t not in bank:
                bank[-1] = t
        print(kind, "bank", len(bank), flush=True)
        t0 = time.time()
        emb = model.encode(bank, batch_size=128, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False, device=device).astype(np.float32)
        qs = [r["query"] for r in rows]
        qe = model.encode([PREFIX + q for q in qs], batch_size=64, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False, device=device).astype(np.float32)
        sims = qe @ emb.T
        ids = np.argsort(-sims, axis=1)[:, :50]
        ranks = []
        for i, r in enumerate(rows):
            found = np.nonzero(np.array([bank[int(j)] == r["target"] for j in ids[i]]))[0]
            ranks.append(int(found[0]) + 1 if len(found) else 0)
        ranks = np.array(ranks)
        rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in (1, 10, 50)}
        mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
        out[kind] = {"n": len(rows), "bank": len(bank), "mrr": mrr, **rec,
                     "elapsed_s": round(time.time() - t0, 1)}
        print(kind, out[kind], flush=True)
    json.dump(out, open(os.path.join(WORK, "semantic_message_metrics.json"), "w"), indent=2)
    print("wrote", os.path.join(WORK, "semantic_message_metrics.json"), flush=True)

if __name__ == "__main__":
    main()
