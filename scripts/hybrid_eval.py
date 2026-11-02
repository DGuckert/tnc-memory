import json, os, re, math, time
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 10485760
CHUNK = 128
K = 50
NF = 2 ** 20

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
    with open(os.path.join(WORK, f"mod_texts_{N}.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            docs[o["pos"]] = o["text"]

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
    word_bytes = n_postings * 5 + len(postings) * 24
    print(f"word bm25 vocab={len(postings)} postings={n_postings} "
          f"est={word_bytes/1e6:.1f}MB ({word_bytes/(n_chunks*CHUNK):.2f} B/token) "
          f"in {time.time()-t0:.1f}s", flush=True)

    k1, b = 1.2, 0.75
    def bm25_scores(q):
        scores = np.zeros(n_chunks, dtype=np.float32)
        for t in set(tokenize(q)):
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
        return idx[np.argsort(-scores[idx])]

    char_vec = HashingVectorizer(n_features=NF, analyzer="char", ngram_range=(2, 3),
                                 lowercase=True, alternate_sign=False, norm=None,
                                 dtype=np.float32)
    X = char_vec.transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    X = tfidf.fit_transform(X)
    char_bytes = X.nnz * 8 + NF * 4
    print(f"char tfidf nnz={X.nnz} est={char_bytes/1e6:.1f}MB "
          f"({char_bytes/(n_chunks*CHUNK):.2f} B/token) in {time.time()-t0:.1f}s", flush=True)

    def char_topk(q):
        qv = tfidf.transform(char_vec.transform([q]))
        s = (X @ qv.T).toarray().ravel()
        return topk(s)

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
        return ranks

    def rrf(a, b, wa=1.0, wb=1.0):
        scores = {}
        for rank, doc in enumerate(a):
            scores[int(doc)] = scores.get(int(doc), 0.0) + wa / (60 + rank + 1)
        for rank, doc in enumerate(b):
            scores[int(doc)] = scores.get(int(doc), 0.0) + wb / (60 + rank + 1)
        order = sorted(scores.items(), key=lambda kv: -kv[1])[:K]
        return np.array([d for d, _ in order], dtype=np.int64)

    # exact and typo conditions
    word_exact, char_exact, rrf_exact, route_exact = [], [], [], []
    word_typo, char_typo, rrf_typo, route_typo = [], [], [], []
    for i, q in enumerate(qtexts):
        name = facts[i]["name"]
        typo = (name[:-2] + name[-1] + name[-2]) if len(name) > 3 else (name + "x")
        q_typo = q.replace(name, typo)
        we = topk(bm25_scores(q)); ce = char_topk(q)
        wt = topk(bm25_scores(q_typo)); ct = char_topk(q_typo)
        word_exact.append(we); char_exact.append(ce)
        word_typo.append(wt); char_typo.append(ct)
        rrf_exact.append(rrf(we, ce)); rrf_typo.append(rrf(wt, ct))
        # Router: exact rare name -> word BM25; otherwise char n-gram.
        df_name = len(postings.get(name.lower(), []))
        route_exact.append(we if df_name > 0 else ce)
        df_typo = len(postings.get(typo.lower(), []))
        route_typo.append(wt if df_typo > 0 else ct)

    noname_word, noname_char = [], []
    for f in facts:
        q = f"what is the {f['noun']} for the project?"
        noname_word.append(topk(bm25_scores(q)))
        noname_char.append(char_topk(q))

    r = {}
    r["word_exact"] = evaluate(word_exact, "word_exact")
    r["word_typo"] = evaluate(word_typo, "word_typo")
    r["char_exact"] = evaluate(char_exact, "char_exact")
    r["char_typo"] = evaluate(char_typo, "char_typo")
    r["rrf_exact"] = evaluate(rrf_exact, "rrf_exact")
    r["rrf_typo"] = evaluate(rrf_typo, "rrf_typo")
    r["route_exact"] = evaluate(route_exact, "route_exact")
    r["route_typo"] = evaluate(route_typo, "route_typo")
    r["word_noname"] = evaluate(noname_word, "word_noname")
    r["char_noname"] = evaluate(noname_char, "char_noname")

    out = {
        "N": N, "n_chunks": n_chunks, "n_queries": len(facts),
        "word_est_bytes": int(word_bytes),
        "word_est_bytes_per_token": float(word_bytes / (n_chunks * CHUNK)),
        "char_est_bytes": int(char_bytes),
        "char_est_bytes_per_token": float(char_bytes / (n_chunks * CHUNK)),
        "combined_est_bytes_per_token": float((word_bytes + char_bytes) / (n_chunks * CHUNK)),
        "conditions": {k: {"recall@1": float(np.mean((v > 0) & (v <= 1))),
                            "recall@10": float(np.mean((v > 0) & (v <= 10))),
                            "recall@50": float(np.mean((v > 0) & (v <= 50))),
                            "ranks": [int(x) for x in v]} for k, v in r.items()},
    }
    json.dump(out, open(os.path.join(WORK, "hybrid_results.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "hybrid_results.json"))

if __name__ == "__main__":
    main()
