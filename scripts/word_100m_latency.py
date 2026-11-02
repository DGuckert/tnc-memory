import json, os, re, math, time
from collections import defaultdict
import numpy as np

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 100_000_000
CHUNK = 128
K = 50
NQ = 50
STOP_DF = 20000

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def main():
    meta = json.load(open(os.path.join(WORK, "needles_100000000.json"), encoding="utf-8"))
    n_chunks = meta["n_chunks"]
    facts = meta["facts"]
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        base_docs = [json.loads(line)["text"] for line in f]
    docs = [base_docs[i % len(base_docs)] for i in range(n_chunks)]
    with open(os.path.join(WORK, "mod_texts_100000000.jsonl"), encoding="utf-8") as f:
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
    print(f"build {time.time()-t0:.1f}s vocab={len(postings)}", flush=True)

    k1, b = 1.2, 0.75
    def scores(q):
        s = np.zeros(n_chunks, dtype=np.float32)
        for t in set(tokenize(q)):
            p = postings.get(t)
            if not p or len(p) > STOP_DF:
                continue
            idf_t = idf[t]
            for i, c in p:
                denom = c + k1 * (1.0 - b + b * doc_len[i] / avgdl)
                s[i] += idf_t * c * (k1 + 1.0) / denom
        return s

    def topk(s):
        idx = np.argpartition(-s, K - 1)[:K]
        return idx[np.argsort(-s[idx])]

    for label, qs in [("exact", [f["query"] for f in facts[:NQ]]),
                      ("typo", [(f["query"].replace(f["name"], f["name"][:-2] + f["name"][-1] + f["name"][-2]) if len(f["name"]) > 3 else f["query"]) for f in facts[:NQ]])]:
        ts = []
        for q in qs:
            t = time.perf_counter()
            topk(scores(q))
            ts.append(time.perf_counter() - t)
        ts = np.array(ts)
        print(f"{label} median_ms={np.median(ts)*1000:.3f} p95_ms={np.percentile(ts,95)*1000:.3f} "
              f"mean_ms={ts.mean()*1000:.3f}", flush=True)

if __name__ == "__main__":
    main()
