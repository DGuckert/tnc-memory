import json, os, time, re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 10485760
CHUNK = 128
K = 50
NF = 2 ** 20

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
    vec = HashingVectorizer(n_features=NF, analyzer="char", ngram_range=(2, 3),
                            lowercase=True, alternate_sign=False, norm=None,
                            dtype=np.float32)
    X = vec.transform(docs)
    print("hashed", X.shape, "nnz", X.nnz, f"in {time.time()-t0:.1f}s", flush=True)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    X = tfidf.fit_transform(X)
    print("tfidf done", f"in {time.time()-t0:.1f}s", flush=True)

    def topk_for(q):
        qv = tfidf.transform(vec.transform([q]))
        s = (X @ qv.T).toarray().ravel()
        idx = np.argpartition(-s, K - 1)[:K]
        return idx[np.argsort(-s[idx])]

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

    exact_ids = []
    typo_ids = []
    for i, q in enumerate(qtexts):
        exact_ids.append(topk_for(q))
        name = facts[i]["name"]
        typo = (name[:-2] + name[-1] + name[-2]) if len(name) > 3 else (name + "x")
        typo_ids.append(topk_for(q.replace(name, typo)))

    r_exact = evaluate(exact_ids, "char_exact")
    r_typo = evaluate(typo_ids, "char_typo")

    out = {
        "N": N, "n_chunks": n_chunks, "n_queries": len(facts),
        "n_features": NF, "ngram_range": [2, 3],
        "nnz": int(X.nnz),
        "est_bytes": int(X.nnz * 8 + NF * 4),
        "est_bytes_per_token": float((X.nnz * 8 + NF * 4) / (n_chunks * CHUNK)),
        "exact": {"recall@1": float(np.mean((r_exact > 0) & (r_exact <= 1))),
                  "recall@10": float(np.mean((r_exact > 0) & (r_exact <= 10))),
                  "recall@50": float(np.mean((r_exact > 0) & (r_exact <= 50))),
                  "ranks": [int(x) for x in r_exact]},
        "typo": {"recall@1": float(np.mean((r_typo > 0) & (r_typo <= 1))),
                 "recall@10": float(np.mean((r_typo > 0) & (r_typo <= 10))),
                 "recall@50": float(np.mean((r_typo > 0) & (r_typo <= 50))),
                 "ranks": [int(x) for x in r_typo]},
    }
    json.dump(out, open(os.path.join(WORK, "char_ngram_results.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "char_ngram_results.json"))

if __name__ == "__main__":
    main()
