import json, os, re, math, time
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 100_000_000
CHUNK = 128
K = 50
NF = 2 ** 20
NQ = int(os.environ.get("NQ", "50"))
STOP_DF = int(os.environ.get("STOP_DF", "20000"))

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def main():
    t_all = time.time()
    meta = json.load(open(os.path.join(WORK, "needles_100000000.json"), encoding="utf-8"))
    n_chunks = meta["n_chunks"]
    gold = np.array(meta["gold_positions"], dtype=np.int64)
    facts = meta["facts"]
    assert n_chunks == N // CHUNK

    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        base_docs = [json.loads(line)["text"] for line in f]
    docs = [base_docs[i % len(base_docs)] for i in range(n_chunks)]
    with open(os.path.join(WORK, "mod_texts_100000000.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            docs[o["pos"]] = o["text"]
    print(f"docs={len(docs)} base={len(base_docs)} in {time.time()-t_all:.1f}s", flush=True)

    pos_noun = {}
    for i, f in enumerate(facts):
        pos_noun[int(gold[i])] = f["noun"]
        for p in meta["distractor_positions"][i]:
            pos_noun[int(p)] = f["noun"]
    noun_to_pos = defaultdict(set)
    for p, noun in pos_noun.items():
        noun_to_pos[noun].add(p)
    same_noun = [noun_to_pos[f["noun"]] - {int(gold[i])} for i, f in enumerate(facts)]

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
    n_postings = sum(len(p) for p in postings.values())
    word_bytes = n_postings * 5 + len(postings) * 24
    print(f"word bm25 vocab={len(postings)} postings={n_postings} "
          f"est={word_bytes/1e6:.1f}MB ({word_bytes/(n_chunks*CHUNK):.2f} B/token) "
          f"build={time.time()-t0:.1f}s", flush=True)

    k1, b = 1.2, 0.75
    def bm25_scores(q):
        scores = np.zeros(n_chunks, dtype=np.float32)
        for t in set(tokenize(q)):
            p = postings.get(t)
            if not p or len(p) > STOP_DF:
                continue
            idf_t = idf[t]
            for i, c in p:
                denom = c + k1 * (1.0 - b + b * doc_len[i] / avgdl)
                scores[i] += idf_t * c * (k1 + 1.0) / denom
        return scores

    def topk(scores, k=K):
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        return idx[np.argsort(-scores[idx])]

    t0 = time.time()
    char_vec = HashingVectorizer(n_features=NF, analyzer="char", ngram_range=(2, 3),
                                 lowercase=True, alternate_sign=False, norm=None,
                                 dtype=np.float32)
    X = char_vec.transform(docs)
    print(f"char hashed shape={X.shape} nnz={X.nnz} in {time.time()-t0:.1f}s", flush=True)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    X = tfidf.fit_transform(X)
    char_bytes = X.nnz * 8 + NF * 4
    print(f"char tfidf est={char_bytes/1e6:.1f}MB "
          f"({char_bytes/(n_chunks*CHUNK):.2f} B/token) total={time.time()-t0:.1f}s", flush=True)

    def char_topk(q):
        qv = tfidf.transform(char_vec.transform([q]))
        s = (X @ qv.T).toarray().ravel()
        return topk(s)

    def evaluate(ids_by_q, label):
        ranks = np.zeros(len(ids_by_q), dtype=np.int32)
        top1_same = 0
        top1_dist = 0
        for i in range(len(ids_by_q)):
            ids = ids_by_q[i]
            hits = np.nonzero(ids == gold[i])[0]
            if len(hits):
                ranks[i] = int(hits[0]) + 1
            if len(ids) and int(ids[0]) in same_noun[i]:
                top1_same += 1
            if len(ids) and int(ids[0]) in meta["distractor_positions"][i]:
                top1_dist += 1
        rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in [1, 10, 50]}
        print(f"{label:16s} " + " ".join(f"{k}={v:.3f}" for k, v in rec.items()) +
              f" same_noun_top1={top1_same/len(ids_by_q):.3f} "
              f"distractor_top1={top1_dist/len(ids_by_q):.3f}", flush=True)
        return {"ranks": [int(x) for x in ranks], **rec,
                "same_noun_top1": top1_same / len(ids_by_q),
                "distractor_top1": top1_dist / len(ids_by_q)}

    word_exact, word_typo, char_exact, char_typo = [], [], [], []
    t0 = time.time()
    for i, f in enumerate(facts[:NQ]):
        q = f["query"]
        name = f["name"]
        typo = (name[:-2] + name[-1] + name[-2]) if len(name) > 3 else (name + "x")
        word_exact.append(topk(bm25_scores(q)))
        word_typo.append(topk(bm25_scores(q.replace(name, typo))))
        char_exact.append(char_topk(q))
        char_typo.append(char_topk(q.replace(name, typo)))
    print(f"queries={NQ} query_time={time.time()-t0:.1f}s", flush=True)

    out = {
        "N": N, "n_chunks": n_chunks, "content_tokens": n_chunks * CHUNK,
        "n_queries": NQ, "stop_df": STOP_DF,
        "word_est_bytes": int(word_bytes),
        "word_est_bytes_per_token": float(word_bytes / (n_chunks * CHUNK)),
        "char_est_bytes": int(char_bytes),
        "char_est_bytes_per_token": float(char_bytes / (n_chunks * CHUNK)),
        "combined_est_bytes_per_token": float((word_bytes + char_bytes) / (n_chunks * CHUNK)),
        "base_est_bytes_per_token": 4.0,
        "total_est_bytes_per_token": float((word_bytes + char_bytes) / (n_chunks * CHUNK) + 4.0),
        "conditions": {
            "word_exact": evaluate(word_exact, "word_exact"),
            "word_typo": evaluate(word_typo, "word_typo"),
            "char_exact": evaluate(char_exact, "char_exact"),
            "char_typo": evaluate(char_typo, "char_typo"),
        },
        "note": "100M-token scale test: 9.75M corpus repeated to 781,250 chunks; 800 facts injected once.",
    }
    out["router_exact"] = out["conditions"]["word_exact"]
    out["router_typo"] = out["conditions"]["char_typo"]
    json.dump(out, open(os.path.join(WORK, "hybrid_results_100m.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "hybrid_results_100m.json"), flush=True)

if __name__ == "__main__":
    main()
