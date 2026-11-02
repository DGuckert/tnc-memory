import json, os, time
import numpy as np
import faiss

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
CHUNK = 128
D = 384
SIZES = [131072, 1048576, 10485760]
K = 50
NQ = 200

def main():
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        n_total = sum(1 for _ in f)
    emb = np.load(os.path.join(WORK, "embeddings_fp32.npy"))
    q = np.load(os.path.join(WORK, "query_vectors.npy")).astype(np.float32)
    out = {"metric": "L2 on normalized embeddings", "n_queries": NQ, "stores": {}}
    for N in SIZES:
        key = str(N)
        n_chunks = min(N // CHUNK, n_total)
        with open(os.path.join(WORK, f"needles_{N}.json")) as fh:
            meta = json.load(fh)
        mod_pos = np.array(meta["modified_positions"], dtype=np.int64)
        mod_vec = np.load(os.path.join(WORK, f"mod_vectors_{N}.npy")).astype(np.float32)
        store = emb[:n_chunks].copy()
        store[mod_pos] = mod_vec
        gold = np.array(meta["gold_positions"], dtype=np.int64)
        recs = {}
        for m in [8, 16, 32, 48]:
            for n_train_req in [8192, 32768]:
                n_train = min(n_train_req, n_chunks)
                for side in ["first", "last"]:
                    train = store[:n_train] if side == "first" else store[-n_train:]
                    t0 = time.perf_counter()
                    index = faiss.IndexPQ(D, m, 8, faiss.METRIC_L2)
                    index.train(train)
                    index.add(store)
                    _, ids = index.search(q, K)
                    ranks = np.zeros(NQ, dtype=np.int32)
                    for i in range(NQ):
                        hits = np.nonzero(ids[i] == gold[i])[0]
                        if len(hits):
                            ranks[i] = int(hits[0]) + 1
                    rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in [1, 10, 50]}
                    rec["build_s"] = time.perf_counter() - t0
                    recs[f"pq{m}_{side}_{n_train_req}"] = rec
                    print(N, m, side, n_train_req, rec, flush=True)
        out["stores"][key] = {"n_chunks": n_chunks, "content_tokens": n_chunks * CHUNK, "configs": recs}
        with open(os.path.join(WORK, "drift_true.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    print("done")

if __name__ == "__main__":
    main()
