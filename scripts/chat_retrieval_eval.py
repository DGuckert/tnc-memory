import json, os, re, math, time
from collections import defaultdict
import numpy as np

BASE = "/home/daniel/tnc"
CORPUS = os.path.join(BASE, "corpus.jsonl")
WORK = os.path.join(BASE, "work", "chat_coding")
K = 50

def tokenize(s):
    return re.findall(r"[a-z0-9_]+", s.lower())

def topk(s, k=K):
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])]

def main():
    docs = []
    t0 = time.time()
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            text = re.sub(r"\s+", " ", o["t"]).strip()
            if len(text) >= 8:
                docs.append(text)
    n = len(docs)
    doc_len = np.zeros(n, dtype=np.float32)
    postings = defaultdict(list)
    for i, d in enumerate(docs):
        toks = tokenize(d)
        doc_len[i] = len(toks)
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, c in tf.items():
            postings[t].append((i, c))
    avgdl = float(doc_len.mean())
    df = {t: len(p) for t, p in postings.items()}
    idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    print("docs", n, "build_s", round(time.time() - t0, 1), flush=True)
    k1, b = 1.2, 0.75
    def search(q):
        s = np.zeros(n, dtype=np.float32)
        for t in set(tokenize(q)):
            p = postings.get(t)
            if not p:
                continue
            idt = idf[t]
            for i, c in p:
                denom = c + k1 * (1.0 - b + b * doc_len[i] / avgdl)
                s[i] += idt * c * (k1 + 1.0) / denom
        return topk(s)

    out = {"n_docs": n}
    for kind in ("coding", "chat"):
        rows = [json.loads(line) for line in open(os.path.join(WORK, f"retrieval_{kind}.jsonl"), encoding="utf-8")]
        rows = [r for r in rows if len(r["target"]) >= 20]
        ranks = []
        examples = []
        t1 = time.time()
        for r in rows:
            ids = search(r["query"])
            hit = np.nonzero(np.array([docs[int(i)] == r["target"] for i in ids]))[0]
            rank = int(hit[0]) + 1 if len(hit) else 0
            ranks.append(rank)
            if len(examples) < 5:
                examples.append({"query": r["query"][:200], "target": r["target"][:200],
                                 "rank": rank, "top1": docs[int(ids[0])][:200]})
        ranks = np.array(ranks)
        rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in (1, 10, 50)}
        mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
        out[kind] = {"n": len(rows), "mrr": mrr, **rec, "elapsed_s": round(time.time() - t1, 1),
                     "examples": examples}
        print(kind, {k: v for k, v in out[kind].items() if k != "examples"}, flush=True)
    json.dump(out, open(os.path.join(WORK, "retrieval_metrics.json"), "w"), indent=2)
    print("wrote", os.path.join(WORK, "retrieval_metrics.json"), flush=True)

if __name__ == "__main__":
    main()
