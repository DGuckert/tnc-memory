import json, os, re, math, difflib
from collections import defaultdict
import numpy as np

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 10485760
CHUNK = 128
K = 50

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def topk(s, k=K):
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])]

def main():
    meta = json.load(open(os.path.join(WORK, f"needles_{N}.json"), encoding="utf-8"))
    n_chunks = meta["n_chunks"]
    gold = np.array(meta["gold_positions"], dtype=np.int64)
    facts = meta["facts"]
    docs = []
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    with open(os.path.join(WORK, f"mod_texts_{N}.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            docs[o["pos"]] = o["text"]

    entities = set()
    for d in docs:
        entities.update(x.lower() for x in re.findall(r"project ([A-Za-z0-9]+) is", d))
    entities = sorted(entities)
    print("entities", len(entities), flush=True)

    doc_len = np.zeros(n_chunks, dtype=np.float32)
    postings = defaultdict(list)
    for i, text in enumerate(docs):
        toks = tokenize(text)
        doc_len[i] = len(toks)
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, c in tf.items():
            postings[t].append((i, c))
    avgdl = float(doc_len.mean())
    df = {t: len(p) for t, p in postings.items()}
    idf = {t: math.log(1.0 + (n_chunks - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    k1, b = 1.2, 0.75
    def scores(q):
        s = np.zeros(n_chunks, dtype=np.float32)
        for t in set(tokenize(q)):
            p = postings.get(t)
            if not p:
                continue
            idt = idf[t]
            for i, c in p:
                denom = c + k1 * (1.0 - b + b * doc_len[i] / avgdl)
                s[i] += idt * c * (k1 + 1.0) / denom
        return s
    def osa(a, b):
        prev2 = None
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
                if prev2 is not None and i > 1 and j > 1 and ca == b[j-2] and a[i-2] == cb:
                    cur[j] = min(cur[j], prev2[j-2] + 1)
            prev2, prev = prev, cur
        return prev[-1]

    def correct(q):
        out = []
        changes = []
        for t in tokenize(q):
            if t in postings or len(t) < 4:
                out.append(t); continue
            cands = sorted((osa(t, e), e) for e in entities)[:3]
            if cands and cands[0][0] <= 2:
                out.append(cands[0][1]); changes.append((t, cands[0][1]))
            else:
                out.append(t)
        return " ".join(out), changes

    def evaluate(qs, label):
        ranks = np.zeros(len(qs), dtype=np.int32)
        ex = []
        for i, q in enumerate(qs):
            q2, ch = correct(q)
            if i < 10: ex.append((q, q2, ch))
            ids = topk(scores(q2))
            hit = np.nonzero(ids == gold[i])[0]
            if len(hit): ranks[i] = int(hit[0]) + 1
        rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in [1, 10, 50]}
        mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
        print(label, rec, "mrr", round(mrr, 3), flush=True)
        return {"ranks": [int(x) for x in ranks], **rec, "mrr": mrr, "examples": ex}

    exact = [f["query"] for f in facts]
    typo = []
    for f in facts:
        n = f["name"]
        v = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        typo.append(f["query"].replace(n, v))
    out = {"N": N, "entities": len(entities)}
    out["exact"] = evaluate(exact, "entity_exact")
    out["typo"] = evaluate(typo, "entity_typo")
    json.dump(out, open(os.path.join(WORK, "entity_fuzzy_results.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "entity_fuzzy_results.json"), flush=True)

if __name__ == "__main__":
    main()
