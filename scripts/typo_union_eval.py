import json, os, re, math, time, difflib
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
RERANKER = "/home/daniel/tnc/training_runs/reranker-mini"

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
    def fuzzy_correct(q):
        out = []
        for t in tokenize(q):
            if t in postings or len(t) < 4:
                out.append(t); continue
            cand = difflib.get_close_matches(t, rare, n=3, cutoff=0.7)
            out.append(cand[0] if cand else t)
        return " ".join(out)

    vec = HashingVectorizer(n_features=NF, analyzer="char", ngram_range=(2, 3),
                            lowercase=True, alternate_sign=False, norm=None, dtype=np.float32)
    X = vec.transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    X = tfidf.fit_transform(X)
    def char_topk(q):
        qv = tfidf.transform(vec.transform([q]))
        return topk((X @ qv.T).toarray().ravel())
    print("indexes", round(time.time() - t0, 1), "rare", len(rare), flush=True)

    tok = AutoTokenizer.from_pretrained(RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER, num_labels=1).to("cuda").eval()
    def rerank(q, ids, k=10):
        passages = [docs[int(i)] for i in ids]
        enc = tok([q] * len(passages), passages, truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            scores = model(**enc).logits.view(-1).float().cpu().numpy()
        order = np.argsort(-scores)[:k]
        return [int(ids[i]) for i in order]

    typo = []
    for f in facts:
        n = f["name"]
        v = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        typo.append(f["query"].replace(n, v))

    stats = {"n": len(facts), "char_gold": 0, "word_gold": 0, "union_gold": 0,
             "rerank1": 0, "rerank5": 0, "rerank10": 0, "example": None}
    for i, q in enumerate(typo):
        cid = char_topk(q)
        wid = topk(bm25_scores(fuzzy_correct(q)))
        union = list(dict.fromkeys([int(x) for x in wid] + [int(x) for x in cid]))
        g = int(gold[i])
        stats["char_gold"] += int(g in set(int(x) for x in cid))
        stats["word_gold"] += int(g in set(int(x) for x in wid))
        stats["union_gold"] += int(g in set(union))
        r = rerank(q, np.array(union, dtype=np.int64), 10)
        stats["rerank1"] += int(g == r[0])
        stats["rerank5"] += int(g in r[:5])
        stats["rerank10"] += int(g in r[:10])
        if stats["example"] is None and g not in r[:5]:
            stats["example"] = {"query": q, "corrected": fuzzy_correct(q), "gold": g,
                                "top": r[:5], "in_char": int(g) in set(int(x) for x in cid),
                                "in_word": int(g) in set(int(x) for x in wid)}
    n = len(facts)
    out = {k: (v / n if isinstance(v, int) and k not in ("n",) else v) for k, v in stats.items() if k != "example"}
    out["example"] = stats["example"]
    print(json.dumps(out, indent=2), flush=True)
    json.dump(out, open(os.path.join(WORK, "typo_union_results.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
