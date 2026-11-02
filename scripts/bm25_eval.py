import json, os, re, math, time
from collections import defaultdict
import numpy as np
import faiss

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 10485760
CHUNK = 128
K = 50

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def main():
    meta = json.load(open(os.path.join(WORK, f"needles_{N}.json"), encoding="utf-8"))
    n_chunks = meta["n_chunks"]
    gold = np.array(meta["gold_positions"], dtype=np.int64)
    facts = meta["facts"]
    qtexts = [f["query"] for f in facts]

    docs = []
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    assert len(docs) == n_chunks, (len(docs), n_chunks)
    with open(os.path.join(WORK, f"mod_texts_{N}.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            docs[o["pos"]] = o["text"]

    t0 = time.time()
    tokenized = [tokenize(d) for d in docs]
    doc_len = np.array([len(t) for t in tokenized], dtype=np.float32)
    avgdl = float(doc_len.mean())
    postings = defaultdict(list)
    for i, toks in enumerate(tokenized):
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, c in tf.items():
            postings[t].append((i, c))
    df = {t: len(p) for t, p in postings.items()}
    idf = {t: math.log(1.0 + (n_chunks - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    n_postings = sum(len(p) for p in postings.values())
    bm25_bytes = n_postings * 5 + len(postings) * 24
    print(f"bm25 built docs={n_chunks} vocab={len(postings)} postings={n_postings} "
          f"avgdl={avgdl:.1f} est={bm25_bytes/1e6:.1f}MB "
          f"({bm25_bytes/(n_chunks*CHUNK):.2f} B/token) in {time.time()-t0:.1f}s", flush=True)

    k1, b = 1.2, 0.75
    def bm25_scores(q):
        scores = np.zeros(n_chunks, dtype=np.float32)
        terms = set(tokenize(q))
        for t in terms:
            p = postings.get(t)
            if not p:
                continue
            idf_t = idf[t]
            for i, c in p:
                denom = c + k1 * (1.0 - b + b * doc_len[i] / avgdl)
                scores[i] += idf_t * c * (k1 + 1.0) / denom
        return scores

    def topk(scores, k=K):
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx

    # same-noun pools: all injected facts with the same noun (gold + designated distractors)
    pos_noun = {}
    for i, f in enumerate(facts):
        pos_noun[int(gold[i])] = f["noun"]
        for p in meta["distractor_positions"][i]:
            pos_noun[int(p)] = f["noun"]
    noun_to_pos = defaultdict(set)
    for p, noun in pos_noun.items():
        noun_to_pos[noun].add(p)
    same_noun = []
    for i, f in enumerate(facts):
        same_noun.append(noun_to_pos[f["noun"]] - {int(gold[i])})

    # dense baseline + hybrid
    emb = np.load(os.path.join(WORK, "embeddings_fp32.npy")).astype(np.float32)
    qv = np.load(os.path.join(WORK, "query_vectors.npy")).astype(np.float32)
    mod_pos = np.array(meta["modified_positions"], dtype=np.int64)
    mod_vec = np.load(os.path.join(WORK, f"mod_vectors_{N}.npy")).astype(np.float32)
    store = emb[:n_chunks].copy()
    store[mod_pos] = mod_vec
    index = faiss.IndexFlatL2(store.shape[1])
    index.add(store)
    _, dense_ids = index.search(qv, K)
    print("dense search done", flush=True)

    def evaluate(ids_by_q, label):
        ranks = np.zeros(len(facts), dtype=np.int32)
        top1_same = 0
        top1_dist = 0
        for i in range(len(facts)):
            ids = ids_by_q[i]
            hits = np.nonzero(ids == gold[i])[0]
            if len(hits):
                ranks[i] = int(hits[0]) + 1
            if len(ids) and int(ids[0]) in same_noun[i]:
                top1_same += 1
            if len(ids) and int(ids[0]) in meta["distractor_positions"][i]:
                top1_dist += 1
        rec = {f"r@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in [1, 10, 50]}
        print(f"{label:18s} " + " ".join(f"{k}={v:.3f}" for k, v in rec.items()) +
              f" same_noun_top1={top1_same/len(facts):.3f} distractor_top1={top1_dist/len(facts):.3f}",
              flush=True)
        return rec, ranks

    bm_ids = []
    bm_scores = []
    for i, q in enumerate(qtexts):
        s = bm25_scores(q)
        bm_scores.append(s)
        bm_ids.append(topk(s))
    evaluate(bm_ids, "bm25")
    evaluate([np.array(x) for x in dense_ids], "dense_flat")

    typo_ids = []
    for i, q in enumerate(qtexts):
        name = facts[i]["name"]
        typo = (name[:-2] + name[-1] + name[-2]) if len(name) > 3 else (name + "x")
        s = bm25_scores(q.replace(name, typo))
        typo_ids.append(topk(s))
    evaluate(typo_ids, "bm25_typo")

    # RRF fusion
    rrf_ids = []
    for i in range(len(facts)):
        scores = {}
        for rank, doc in enumerate(dense_ids[i]):
            scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (60 + rank + 1)
        for rank, doc in enumerate(bm_ids[i]):
            scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (60 + rank + 1)
        order = sorted(scores.items(), key=lambda kv: -kv[1])[:K]
        rrf_ids.append(np.array([d for d, _ in order], dtype=np.int64))
    evaluate(rrf_ids, "rrf_dense_bm25")

    # Dense + BM25 candidate union, then BM25 score tie-break for candidates
    union_ids = []
    for i in range(len(facts)):
        cand = list(dict.fromkeys([int(x) for x in dense_ids[i]] + [int(x) for x in bm_ids[i]]))
        cand = sorted(cand, key=lambda d: -float(bm_scores[i][d]))[:K]
        union_ids.append(np.array(cand, dtype=np.int64))
    evaluate(union_ids, "bm25_on_union")

    # save
    out = {"N": N, "n_chunks": n_chunks, "n_queries": len(facts),
           "bm25_est_bytes": int(bm25_bytes),
           "bm25_est_bytes_per_token": float(bm25_bytes / (n_chunks * CHUNK))}
    for label, ids in [("bm25", bm_ids), ("dense_flat", dense_ids), ("rrf", rrf_ids),
                       ("bm25_union", union_ids), ("bm25_typo", typo_ids)]:
        ranks = []
        for i in range(len(facts)):
            hits = np.nonzero(ids[i] == gold[i])[0]
            ranks.append(int(hits[0]) + 1 if len(hits) else 0)
        out[label] = {
            "recall@1": float(np.mean([1 <= r <= 1 for r in ranks])),
            "recall@10": float(np.mean([1 <= r <= 10 for r in ranks])),
            "recall@50": float(np.mean([1 <= r <= 50 for r in ranks])),
            "ranks": ranks,
        }
    json.dump(out, open(os.path.join(WORK, "bm25_results.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "bm25_results.json"))

if __name__ == "__main__":
    main()
