import json, os, time, statistics
import numpy as np
import faiss

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
CHUNK = 128
SIZES = [131072, 1048576, 10485760]
D = 384
BASE_BYTES_PER_TOKEN = 4.0  # 3 B token id + 1 B blank-run flag
BUDGET = 16.0
PQ_CODEBOOK_BYTES = 256 * D * 4  # 393216 B per PQ index (all m)
SQ8_PARAM_BYTES = D * 2 * 4     # 3072 B trained min/max params
K = 50
NQ = 200

def build_config(name, store, n_chunks):
    t0 = time.perf_counter()
    if name == "flat_fp32":
        index = faiss.IndexFlatL2(D)
        index.add(store)
        bytes_per_chunk = D * 4
        codebook = 0
    elif name == "flat_fp16":
        index = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
        index.train(store)
        index.add(store)
        bytes_per_chunk = D * 2
        codebook = 0
    elif name == "flat_int8":
        index = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        index.train(store)
        index.add(store)
        bytes_per_chunk = D * 1
        codebook = SQ8_PARAM_BYTES
    elif name.startswith("pq"):
        parts = name.split("_")
        m = int(parts[0][2:])
        train = parts[1]
        if train == "early":
            n_train = min(8192, n_chunks)
            train_vecs = store[:n_train]
        else:
            n_train = min(32768, n_chunks)
            if n_train == n_chunks:
                train_vecs = store
            else:
                rng = np.random.default_rng(777)
                idx = rng.choice(n_chunks, size=n_train, replace=False)
                train_vecs = store[np.sort(idx)]
        index = faiss.IndexPQ(D, m, 8, faiss.METRIC_L2)
        index.train(train_vecs)
        index.add(store)
        bytes_per_chunk = m
        codebook = PQ_CODEBOOK_BYTES
    else:
        raise ValueError(name)
    build_s = time.perf_counter() - t0
    return index, bytes_per_chunk, codebook, build_s

def measure(index, q):
    for _ in range(3):
        index.search(q[:8], K)
    t1 = []
    for i in range(len(q)):
        t = time.perf_counter()
        index.search(q[i:i+1], K)
        t1.append(time.perf_counter() - t)
    t200 = []
    for _ in range(5):
        t = time.perf_counter()
        index.search(q, K)
        t200.append(time.perf_counter() - t)
    t1 = np.array(t1)
    t200 = np.array(t200)
    return {
        "batch1_median_ms": float(np.median(t1) * 1000),
        "batch1_mean_ms": float(t1.mean() * 1000),
        "batch1_p95_ms": float(np.percentile(t1, 95) * 1000),
        "batch200_median_ms": float(np.median(t200) * 1000),
        "batch200_qps": float(len(q) / np.median(t200)),
        "batch1_n": len(q),
        "batch200_repeats": 5,
    }

def main():
    with open(os.path.join(WORK, "stats.json")) as f:
        stats = json.load(f)
    with open(os.path.join(WORK, "embed_meta.json")) as f:
        embed_meta = json.load(f)
    with open(os.path.join(WORK, "needle_meta.json")) as f:
        needle_meta = json.load(f)
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    emb = np.load(os.path.join(WORK, "embeddings_fp32.npy"))
    q = np.load(os.path.join(WORK, "query_vectors.npy")).astype(np.float32)
    total_chunks = len(chunks)
    print("loaded", emb.shape, q.shape, "faiss_threads", faiss.omp_get_max_threads(), flush=True)

    names = ["flat_fp32", "flat_fp16", "flat_int8"]
    for m in [8, 16, 32, 48]:
        names += [f"pq{m}_early", f"pq{m}_late"]

    out = {
        "meta": {
            "corpus": stats.get("corpus"),
            "model": embed_meta["model"],
            "chunk_content_tokens": CHUNK,
            "base_bytes_per_token": BASE_BYTES_PER_TOKEN,
            "budget_bytes_per_token": BUDGET,
            "pq_codebook_bytes": PQ_CODEBOOK_BYTES,
            "sq8_param_bytes": SQ8_PARAM_BYTES,
            "faiss_threads": int(faiss.omp_get_max_threads()),
            "recall_k": K,
            "n_queries": NQ,
            "metric": "L2 on normalized embeddings (equivalent ranking to cosine)",
        },
        "tokens": stats,
        "embedding": embed_meta,
        "needles": needle_meta,
        "stores": {},
        "failure_examples": {},
    }

    for N in SIZES:
        key = str(N)
        if key not in needle_meta:
            continue
        n_chunks = min(N // CHUNK, total_chunks)
        with open(os.path.join(WORK, f"needles_{N}.json")) as fh:
            meta = json.load(fh)
        assert meta["n_chunks"] == n_chunks
        mod_pos = np.array(meta["modified_positions"], dtype=np.int64)
        mod_vec = np.load(os.path.join(WORK, f"mod_vectors_{N}.npy")).astype(np.float32)
        store = emb[:n_chunks].copy()
        store[mod_pos] = mod_vec
        content_tokens = n_chunks * CHUNK
        print(f"\n=== N={N} n_chunks={n_chunks} content_tokens={content_tokens} ===", flush=True)

        # load modified texts for examples
        mod_text = {}
        with open(os.path.join(WORK, f"mod_texts_{N}.jsonl"), encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                mod_text[o["pos"]] = o["text"]

        store_out = {
            "N_requested": N,
            "n_chunks": n_chunks,
            "content_tokens": content_tokens,
            "raw_tokens_equivalent": None,
            "configs": {},
        }
        gold = np.array(meta["gold_positions"], dtype=np.int64)
        dist_pos = meta["distractor_positions"]
        ranks_by_config = {}
        search_cache = {}

        for name in names:
            index, bytes_per_chunk, codebook, build_s = build_config(name, store, n_chunks)
            dist, ids = index.search(q, K)
            search_cache[name] = (dist.copy(), ids.copy())
            ranks = np.zeros(NQ, dtype=np.int32)
            top1 = ids[:, 0].copy()
            for i in range(NQ):
                hits = np.nonzero(ids[i] == gold[i])[0]
                if len(hits):
                    ranks[i] = int(hits[0]) + 1
            rec = {f"recall@{k}": float(np.mean((ranks > 0) & (ranks <= k))) for k in [1, 10, 50]}
            mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
            d1 = sum(1 for i in range(NQ) if int(top1[i]) in dist_pos[i]) / NQ
            d10 = sum(1 for i in range(NQ) if any(int(x) in dist_pos[i] for x in ids[i, :10])) / NQ
            lat = measure(index, q)
            locator_bpt = (n_chunks * bytes_per_chunk + codebook) / content_tokens
            total_bpt = BASE_BYTES_PER_TOKEN + locator_bpt
            cfg = {
                **rec,
                "mrr@50": mrr,
                "distractor_top1_rate": float(d1),
                "distractor_top10_rate": float(d10),
                "ranks": [int(x) for x in ranks],
                "bytes_per_chunk": int(bytes_per_chunk),
                "codebook_bytes": int(codebook),
                "index_payload_bytes": int(n_chunks * bytes_per_chunk),
                "locator_bytes_per_token": float(locator_bpt),
                "total_bytes_per_token": float(total_bpt),
                "headroom_bytes_per_token": float(BUDGET - total_bpt),
                "fits_budget": bool(total_bpt <= BUDGET + 1e-9),
                "build_s": float(build_s),
                "latency": lat,
            }
            store_out["configs"][name] = cfg
            ranks_by_config[name] = ranks
            print(f"{name:14s} r@1={rec['recall@1']:.3f} r@10={rec['recall@10']:.3f} "
                  f"r@50={rec['recall@50']:.3f} bpt={total_bpt:.3f} "
                  f"b1={lat['batch1_median_ms']:.2f}ms b200={lat['batch200_median_ms']:.2f}ms", flush=True)

        # failure examples at this store size
        examples = {}
        for name in ["flat_fp32", "flat_fp16", "flat_int8", "pq8_early", "pq48_early", "pq48_late"]:
            dist, ids = search_cache[name]
            rows = []
            for i in range(NQ):
                r = int(ranks_by_config[name][i])
                if r == 0 or r > 10:
                    gp = int(gold[i])
                    top = int(ids[i, 0])
                    rows.append({
                        "query_i": i,
                        "query": meta["facts"][i]["query"],
                        "noun": meta["facts"][i]["noun"],
                        "project": meta["facts"][i]["name"],
                        "gold_pos": gp,
                        "gold_rank": r,
                        "gold_text": mod_text.get(gp, chunks[gp]["text"])[:500],
                        "top1_pos": top,
                        "top1_text": mod_text.get(top, chunks[top]["text"])[:500],
                        "top1_is_distractor": bool(top in dist_pos[i]),
                        "top1_dist": float(dist[i, 0]),
                    })
                if len(rows) >= 10:
                    break
            examples[name] = rows
        # drift examples: late finds, early misses
        drift = []
        for i in range(NQ):
            re_ = int(ranks_by_config["pq48_early"][i])
            rl = int(ranks_by_config["pq48_late"][i])
            if (re_ == 0 or re_ > 10) and rl > 0 and rl <= 10:
                gp = int(gold[i])
                drift.append({
                    "query_i": i,
                    "query": meta["facts"][i]["query"],
                    "gold_pos": gp,
                    "early_rank": re_,
                    "late_rank": rl,
                    "gold_text": mod_text.get(gp, chunks[gp]["text"])[:500],
                })
            if len(drift) >= 10:
                break
        examples["pq48_drift_late_finds_early_misses"] = drift
        out["failure_examples"][key] = examples
        out["stores"][key] = store_out

    # latency extrapolation from the largest store
    max_key = max(out["stores"].keys(), key=lambda k: out["stores"][k]["content_tokens"])
    T = out["stores"][max_key]["content_tokens"]
    ext = {"basis_N": int(max_key), "basis_content_tokens": T,
           "method": "linear in store size (chunks), measured retrieval-only; marked extrapolation",
           "targets": {}}
    for target in [100_000_000, 500_000_000]:
        ext["targets"][str(target)] = {}
        for name, cfg in out["stores"][max_key]["configs"].items():
            scale = target / T
            ext["targets"][str(target)][name] = {
                "batch1_median_ms": cfg["latency"]["batch1_median_ms"] * scale,
                "batch200_median_ms": cfg["latency"]["batch200_median_ms"] * scale,
                "index_payload_bytes": cfg["index_payload_bytes"] * scale,
                "total_bytes_per_token": cfg["total_bytes_per_token"],
                "extrapolated": True,
            }
    out["latency_extrapolation"] = ext

    with open(os.path.join(WORK, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results.json", flush=True)

if __name__ == "__main__":
    main()
