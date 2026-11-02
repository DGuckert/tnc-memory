import json, os, re, math, time
from collections import defaultdict
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 10485760
CHUNK = 128
K = 50
TOPK = 5
NF = 2 ** 20
RERANKER = "/home/daniel/tnc/training_runs/reranker-mini"
MIDDLEMAN = "/home/daniel/tnc/training_runs/middleman-lora"
BASE_LLM = "HuggingFaceTB/SmolLM2-360M-Instruct"

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def topk(scores, k=K):
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]

def extract_note(text):
    m = re.findall(r"Note to self:.*", text)
    if m:
        return m[-1].strip()
    return text.strip()[-300:]

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
    vec = HashingVectorizer(n_features=NF, analyzer="char", ngram_range=(2, 3),
                            lowercase=True, alternate_sign=False, norm=None, dtype=np.float32)
    X = vec.transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    X = tfidf.fit_transform(X)
    def char_topk(q):
        qv = tfidf.transform(vec.transform([q]))
        return topk((X @ qv.T).toarray().ravel())
    print("indexes", round(time.time() - t0, 1), flush=True)

    rtok = AutoTokenizer.from_pretrained(RERANKER)
    rmodel = AutoModelForSequenceClassification.from_pretrained(RERANKER, num_labels=1).to("cuda").eval()

    mtok = AutoTokenizer.from_pretrained(MIDDLEMAN)
    base = AutoModelForCausalLM.from_pretrained(BASE_LLM, torch_dtype=torch.float16).to("cuda")
    reader = PeftModel.from_pretrained(base, MIDDLEMAN).eval()
    reader.config.use_cache = True
    print("models loaded", flush=True)

    def rerank(query, ids):
        passages = [docs[int(i)] for i in ids]
        enc = rtok([query] * len(passages), passages, truncation=True, max_length=256,
                   padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            scores = rmodel(**enc).logits.view(-1).float().cpu().numpy()
        order = np.argsort(-scores)[:TOPK]
        return [int(ids[i]) for i in order]

    def answer(query, ids):
        notes = [extract_note(docs[int(i)]) for i in ids]
        user = "Memory notes:\n" + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
        user += f"\nQuestion: {query}"
        msgs = [{"role": "system", "content": "You answer only from the memory notes. If the answer is not present, say exactly: I don't know."},
                {"role": "user", "content": user}]
        prompt = mtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = mtok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = reader.generate(**enc, max_new_tokens=16, do_sample=False,
                                  pad_token_id=mtok.eos_token_id, eos_token_id=mtok.eos_token_id)
        return mtok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    exact_q = [f["query"] for f in facts]
    typo_q = []
    for f in facts:
        n = f["name"]
        typo = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        typo_q.append(f["query"].replace(n, typo))

    out = {"N": N, "n_queries": len(facts), "topk": TOPK, "reranker": RERANKER, "middleman": MIDDLEMAN}
    for label, qs, mode in [("exact", exact_q, "word"), ("typo", typo_q, "char")]:
        t1 = time.time()
        gold_top5 = 0; reader_ok = 0; e2e_ok = 0
        examples = []
        for i, q in enumerate(qs):
            ids = topk(bm25_scores(q)) if mode == "word" else char_topk(q)
            ids = rerank(q, ids)
            gold_ok = int(gold[i]) in ids
            if gold_ok:
                gold_top5 += 1
            ans = answer(q, ids)
            if ans == facts[i]["code"]:
                e2e_ok += 1
                if gold_ok:
                    reader_ok += 1
            if len(examples) < 6:
                examples.append({"query": q, "gold": facts[i]["code"], "pred": ans,
                                 "gold_in_top5": gold_ok,
                                 "notes": [extract_note(docs[int(x)]) for x in ids]})
        n = len(qs)
        out[label] = {
            "n": n, "gold_in_top5": gold_top5 / n,
            "reader_given_gold": reader_ok / max(1, gold_top5),
            "end_to_end_exact": e2e_ok / n,
            "elapsed_s": round(time.time() - t1, 1), "examples": examples,
        }
        print(label, {k: v for k, v in out[label].items() if k != "examples"}, flush=True)
    json.dump(out, open(os.path.join(WORK, "end_to_end_975m.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "end_to_end_975m.json"), flush=True)

if __name__ == "__main__":
    main()
