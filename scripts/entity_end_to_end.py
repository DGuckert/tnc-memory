import json, os, re, math, time
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 10485760
CHUNK = 128
K = 50
RERANKER = "/home/daniel/tnc/training_runs/reranker-mini"
MIDDLEMAN = "/home/daniel/tnc/training_runs/middleman-lora"
BASE_LLM = "HuggingFaceTB/SmolLM2-360M-Instruct"

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def topk(s, k=K):
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])]

def extract_note(text):
    m = re.findall(r"Note to self:.*", text)
    return m[-1].strip() if m else text.strip()

def canonical_from_note(note):
    m = re.search(r"the (.+?) for project ([A-Za-z0-9]+) is", note)
    if not m:
        return None
    return f"what is the {m.group(1)} for project {m.group(2)}?"

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

    entities = sorted({x.lower() for d in docs for x in re.findall(r"project ([A-Za-z0-9]+) is", d)})
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
        for t in tokenize(q):
            if t in postings or len(t) < 4:
                out.append(t); continue
            cands = sorted((osa(t, e), e) for e in entities)[:3]
            out.append(cands[0][1] if cands and cands[0][0] <= 2 else t)
        return " ".join(out)
    print("indexes", round(time.time() - t0, 1), "entities", len(entities), flush=True)

    rtok = AutoTokenizer.from_pretrained(RERANKER)
    rmodel = AutoModelForSequenceClassification.from_pretrained(RERANKER, num_labels=1).to("cuda").eval()
    mtok = AutoTokenizer.from_pretrained(MIDDLEMAN)
    base = AutoModelForCausalLM.from_pretrained(BASE_LLM, torch_dtype=torch.float16).to("cuda")
    reader = PeftModel.from_pretrained(base, MIDDLEMAN).eval()
    reader.config.use_cache = True

    def rerank(q, ids, k=5):
        passages = [docs[int(i)] for i in ids]
        enc = rtok([q] * len(passages), passages, truncation=True, max_length=256,
                   padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            sc = rmodel(**enc).logits.view(-1).float().cpu().numpy()
        return [int(ids[i]) for i in np.argsort(-sc)[:k]]
    def answer(q, ids):
        notes = [extract_note(docs[int(i)]) for i in ids]
        user = "Memory notes:\n" + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
        user += f"\nQuestion: {q}"
        msgs = [{"role": "system", "content": "You answer only from the memory notes. If the answer is not present, say exactly: I don't know."},
                {"role": "user", "content": user}]
        prompt = mtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = mtok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = reader.generate(**enc, max_new_tokens=16, do_sample=False,
                                  pad_token_id=mtok.eos_token_id, eos_token_id=mtok.eos_token_id)
        return mtok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    typo = []
    for f in facts:
        n = f["name"]
        v = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        typo.append(f["query"].replace(n, v))

    stats = {"n": len(facts), "pre1": 0, "pre5": 0, "pre10": 0, "pre50": 0,
             "gold_top5": 0, "rerank1": 0, "rerank5": 0, "reader_gold": 0, "e2e": 0}
    examples = []
    t1 = time.time()
    for i, q in enumerate(typo):
        q2 = correct(q)
        ids = topk(scores(q2))
        g = int(gold[i])
        stats["pre1"] += int(len(np.nonzero(ids[:1] == g)[0]) > 0)
        stats["pre5"] += int(g in set(int(x) for x in ids[:5]))
        stats["pre10"] += int(g in set(int(x) for x in ids[:10]))
        stats["pre50"] += int(g in set(int(x) for x in ids))
        canon0 = canonical_from_note(extract_note(docs[ids[0]])) or q2
        top = rerank(canon0, ids, 5)
        gold_ok = g in top
        stats["gold_top5"] += int(gold_ok)
        stats["rerank1"] += int(g == top[0])
        stats["rerank5"] += int(gold_ok)
        canon = canonical_from_note(extract_note(docs[top[0]])) or canon0
        ans = answer(canon, top)
        ok = ans == facts[i]["code"]
        stats["e2e"] += int(ok)
        if ok and gold_ok:
            stats["reader_gold"] += 1
        if len(examples) < 6:
            examples.append({"query": q, "corrected": q2, "gold": facts[i]["code"], "pred": ans,
                             "gold_top5": gold_ok, "top": [extract_note(docs[x]) for x in top]})
    n = len(facts)
    out = {k: (v / n if k not in ("n",) else v) for k, v in stats.items()}
    out["reader_given_gold"] = stats["reader_gold"] / max(1, stats["gold_top5"])
    out["elapsed_s"] = round(time.time() - t1, 1)
    out["examples"] = examples
    json.dump(out, open(os.path.join(WORK, "entity_typo_975m.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
