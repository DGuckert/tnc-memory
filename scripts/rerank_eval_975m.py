import json, os, re, math, time
from collections import defaultdict
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 10485760
CHUNK = 128
K = 50
NF = 2 ** 20
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ADAPTER = "/home/daniel/tnc/training_runs/reranker-mini"

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def topk(scores, k=K):
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]

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
    print("bm25", time.time() - t0, flush=True)

    k1, b = 1.2, 0.75
    def bm25_scores(q):
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

    t0 = time.time()
    vec = HashingVectorizer(n_features=NF, analyzer="char", ngram_range=(2, 3),
                            lowercase=True, alternate_sign=False, norm=None, dtype=np.float32)
    X = vec.transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    X = tfidf.fit_transform(X)
    print("char", time.time() - t0, "nnz", X.nnz, flush=True)

    def char_topk(q):
        qv = tfidf.transform(vec.transform([q]))
        return topk((X @ qv.T).toarray().ravel())

    exact_q = [f["query"] for f in facts]
    typo_q = []
    for f in facts:
        n = f["name"]
        typo = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        typo_q.append(f["query"].replace(n, typo))

    word_ids = [topk(bm25_scores(q)) for q in exact_q]
    char_ids = [char_topk(q) for q in typo_q]
    print("candidates", time.time() - t0, flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1).to("cuda").eval()
    ft = AutoModelForSequenceClassification.from_pretrained(ADAPTER, num_labels=1).to("cuda").eval()

    def rerank(model, qs, cand_ids):
        out = []
        with torch.inference_mode():
            for q, ids in zip(qs, cand_ids):
                passages = [docs[int(i)] for i in ids]
                enc = tok([q] * len(passages), passages, truncation=True, max_length=256,
                          padding=True, return_tensors="pt").to("cuda")
                scores = model(**enc).logits.view(-1).float().cpu().numpy()
                order = np.argsort(-scores)
                out.append(np.array([ids[int(i)] for i in order], dtype=np.int64))
        return out

    def evaluate(ids_by_q, label):
        ranks = np.zeros(len(ids_by_q), dtype=np.int32)
        for i, ids in enumerate(ids_by_q):
            hit = np.nonzero(ids == gold[i])[0]
            if len(hit):
                ranks[i] = int(hit[0]) + 1
        rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in [1, 10, 50]}
        mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
        print(label, rec, "mrr", round(mrr, 3), flush=True)
        return {"ranks": [int(x) for x in ranks], **rec, "mrr": mrr}

    out = {"N": N, "n_queries": len(facts), "k": K,
           "base_model": BASE_MODEL, "adapter": ADAPTER}
    out["word_exact_pre"] = evaluate(word_ids, "word_exact_pre")
    out["char_typo_pre"] = evaluate(char_ids, "char_typo_pre")
    out["word_exact_base"] = evaluate(rerank(base, exact_q, word_ids), "word_exact_base")
    out["word_exact_ft"] = evaluate(rerank(ft, exact_q, word_ids), "word_exact_ft")
    out["char_typo_base"] = evaluate(rerank(base, typo_q, char_ids), "char_typo_base")
    out["char_typo_ft"] = evaluate(rerank(ft, typo_q, char_ids), "char_typo_ft")
    json.dump(out, open(os.path.join(WORK, "reranker_975m.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "reranker_975m.json"), flush=True)

if __name__ == "__main__":
    main()
