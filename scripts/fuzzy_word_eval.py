import json, os, re, math, time, difflib
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

    t0 = time.time()
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
    rare = [t for t, d in df.items() if d <= 20 and len(t) >= 4]
    print("bm25", round(time.time() - t0, 1), "rare", len(rare), flush=True)

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

    def correct(q):
        out = []
        changes = []
        for t in tokenize(q):
            if t in postings or len(t) < 4:
                out.append(t)
                continue
            cand = difflib.get_close_matches(t, rare, n=3, cutoff=0.7)
            if cand:
                out.append(cand[0])
                changes.append((t, cand[0]))
            else:
                out.append(t)
        return " ".join(out), changes

    exact = [f["query"] for f in facts]
    typo = []
    for f in facts:
        n = f["name"]
        v = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        typo.append(f["query"].replace(n, v))

    def evaluate(qs, label):
        ranks = np.zeros(len(qs), dtype=np.int32)
        changes = []
        for i, q in enumerate(qs):
            q2, ch = correct(q)
            if i < 10:
                changes.append((q, q2, ch))
            ids = topk(scores(q2))
            hit = np.nonzero(ids == gold[i])[0]
            if len(hit):
                ranks[i] = int(hit[0]) + 1
        rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in [1, 10, 50]}
        mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
        print(label, rec, "mrr", round(mrr, 3), flush=True)
        return {"ranks": [int(x) for x in ranks], **rec, "mrr": mrr, "examples": changes}

    out = {"N": N, "n_queries": len(facts), "rare_vocab": len(rare)}
    out["exact"] = evaluate(exact, "fuzzy_exact")
    out["typo"] = evaluate(typo, "fuzzy_typo")
    json.dump(out, open(os.path.join(WORK, "fuzzy_word_results.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "fuzzy_word_results.json"), flush=True)

if __name__ == "__main__":
    main()
