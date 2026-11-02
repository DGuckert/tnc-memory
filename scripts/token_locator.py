import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
MODEL = "BAAI/bge-small-en-v1.5"
N = 10485760
CHUNK = 128
NW = 32          # token windows per chunk
PQ_M = 4         # bytes per window
PQ_BITS = 8
TOPK_WINDOWS = 5000
SIZES = [N]

def pool_windows(states, nw=NW):
    L = states.shape[0]
    if L == 0:
        return torch.zeros(nw, states.shape[1], dtype=torch.float32)
    idx = torch.linspace(0, L, nw + 1).round().long()
    out = []
    for k in range(nw):
        a = int(idx[k].item()); b = int(idx[k + 1].item())
        if b <= a:
            b = min(a + 1, L)
        out.append(states[a:b].float().mean(0))
    out = torch.stack(out)
    return F.normalize(out, dim=-1)

def extract(texts, tok, model, batch=64, maxlen=256, nw=NW):
    res = np.zeros((len(texts), nw, 384), dtype=np.float16)
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"]
        for i in range(len(chunk)):
            L = int(mask[i].sum().item())
            st = h[i, 1:L - 1] if L > 2 else h[i, :1]
            res[start + i] = pool_windows(st, nw).cpu().numpy().astype(np.float16)
    return res

def main():
    t0 = time.time()
    chunks = [json.loads(l) for l in open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8")]
    meta = json.load(open(os.path.join(WORK, f"needles_{N}.json")))
    n_chunks = meta["n_chunks"]
    assert n_chunks <= len(chunks)
    print("chunks", len(chunks), "store", n_chunks, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).to("cuda").eval()
    print("model loaded", time.time() - t0, flush=True)

    base_path = os.path.join(WORK, "token_windows_base.npy")
    if os.path.exists(base_path):
        windows = np.load(base_path)
        print("loaded base windows", windows.shape, flush=True)
    else:
        texts = [c["text"] for c in chunks]
        t1 = time.time()
        windows = extract(texts, tok, model, batch=64, maxlen=256, nw=NW)
        n_bad = int(np.sum(~np.isfinite(windows)))
        print(f"base windows {windows.shape} in {time.time()-t1:.1f}s nonfinite={n_bad}", flush=True)
        windows = np.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(base_path, windows)
        print("saved base windows", flush=True)

    mod_pos = np.array(meta["modified_positions"], dtype=np.int64)
    mod_path = os.path.join(WORK, f"token_windows_mod_{N}.npy")
    if os.path.exists(mod_path):
        mw = np.load(mod_path)
        print("loaded modified windows", mw.shape, flush=True)
    else:
        mod_texts = []
        with open(os.path.join(WORK, f"mod_texts_{N}.jsonl"), encoding="utf-8") as f:
            for line in f:
                mod_texts.append(json.loads(line)["text"])
        t2 = time.time()
        mw = extract(mod_texts, tok, model, batch=64, maxlen=256, nw=NW)
        n_bad = int(np.sum(~np.isfinite(mw)))
        print(f"modified windows {mw.shape} in {time.time()-t2:.1f}s nonfinite={n_bad}", flush=True)
        mw = np.nan_to_num(mw, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(mod_path, mw)
    windows[mod_pos] = mw
    print("applied modified windows", flush=True)

    store_w = windows[:n_chunks].reshape(-1, 384)
    n_windows = store_w.shape[0]
    print("store windows", store_w.shape, flush=True)

    rng = np.random.default_rng(0)
    n_sample = min(200000, n_windows)
    sample_idx = rng.choice(n_windows, size=n_sample, replace=False)
    sample = store_w[sample_idx].astype(np.float32)
    index = faiss.IndexPQ(384, PQ_M, PQ_BITS, faiss.METRIC_L2)
    t3 = time.time()
    index.train(sample)
    print(f"PQ trained in {time.time()-t3:.1f}s", flush=True)
    t4 = time.time()
    for start in range(0, n_windows, 200000):
        index.add(store_w[start:start + 200000].astype(np.float32))
    print(f"PQ added {index.ntotal} in {time.time()-t4:.1f}s", flush=True)

    # query token states
    qtexts = [meta["query_prefix"] + f["query"] for f in meta["facts"]]
    enc = tok(qtexts, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
    with torch.no_grad():
        qh = model(**enc).last_hidden_state
    qmask = enc["attention_mask"]
    qlist = []
    for i in range(len(qtexts)):
        L = int(qmask[i].sum().item())
        st = qh[i, 1:L - 1] if L > 2 else qh[i, :1]
        st = F.normalize(st.float(), dim=-1)
        qlist.append(st.cpu().numpy().astype(np.float32))
    print("query token counts", [len(x) for x in qlist[:5]], "mean", np.mean([len(x) for x in qlist]), flush=True)

    gold = np.array(meta["gold_positions"])
    facts = meta["facts"]
    pos_noun = {}
    for i, f in enumerate(facts):
        pos_noun[meta["gold_positions"][i]] = f["noun"]
        for j in range(3):
            pos_noun[meta["distractor_positions"][i][j]] = f["noun"]
    same_noun = {noun: set(p for p, no in pos_noun.items() if no == noun) for noun in set(pos_noun.values())}

    ranks = np.zeros(len(gold), dtype=np.int32)
    top1s = []
    top10s = []
    t5 = time.time()
    for i, qt in enumerate(qlist):
        scores = np.zeros(n_chunks, dtype=np.float32)
        for t in range(qt.shape[0]):
            D, I = index.search(qt[t:t + 1], TOPK_WINDOWS)
            sims = 1.0 - D[0] / 2.0
            best = {}
            for idx, s in zip(I[0], sims):
                c = int(idx) // NW
                if s > best.get(c, -1e9):
                    best[c] = float(s)
            for c, s in best.items():
                scores[c] += s
        order = np.argsort(-scores)
        top1s.append(int(order[0]))
        top10s.append([int(x) for x in order[:10]])
        w = np.where(order == gold[i])[0]
        ranks[i] = int(w[0]) + 1 if len(w) else 0
        if (i + 1) % 20 == 0:
            print("query", i + 1, "elapsed", round(time.time() - t5, 1), flush=True)

    rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in [1, 10, 50]}
    same1 = sum(1 for i in range(len(gold)) if top1s[i] in (same_noun[facts[i]["noun"]] - {int(gold[i])})) / len(gold)
    same10 = sum(1 for i in range(len(gold)) if any(t in (same_noun[facts[i]["noun"]] - {int(gold[i])}) for t in top10s[i])) / len(gold)
    out = {
        "N_requested": N,
        "n_chunks": n_chunks,
        "content_tokens": n_chunks * CHUNK,
        "method": "32 token-window mean-pooled BGE hidden states per 128-token chunk, PQ m=4 (4 B/window), ColBERT-lite max-sim over query tokens",
        "windows_per_chunk": NW,
        "pq_m_bytes_per_window": PQ_M,
        "bytes_per_chunk": NW * PQ_M,
        "locator_bytes_per_token": NW * PQ_M / CHUNK,
        "base_bytes_per_token": 4.0,
        "total_bytes_per_token": 4.0 + NW * PQ_M / CHUNK + (256 * 384 * 4) / (n_chunks * CHUNK),
        "recall@1": rec["recall@1"],
        "recall@10": rec["recall@10"],
        "recall@50": rec["recall@50"],
        "same_noun_top1_rate": float(same1),
        "same_noun_top10_rate": float(same10),
        "median_rank": int(np.median(ranks)),
        "ranks": [int(x) for x in ranks],
        "search_seconds": time.time() - t5,
    }
    json.dump(out, open(os.path.join(WORK, "token_locator_results.json"), "w"), indent=2)
    print(json.dumps({k: out[k] for k in ["n_chunks", "recall@1", "recall@10", "recall@50",
          "same_noun_top1_rate", "median_rank", "total_bytes_per_token", "search_seconds"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
