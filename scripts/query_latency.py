import json, os, time
import numpy as np
from sentence_transformers import SentenceTransformer

WORK = "/home/daniel/tnc/work"
MODEL = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "

def main():
    data = json.load(open(os.path.join(WORK, "needles_131072.json"), encoding="utf-8"))
    queries = [PREFIX + f["query"] for f in data["facts"]]
    model = SentenceTransformer(MODEL, device="cuda")
    model.max_seq_length = 256
    model.half()
    model.encode(queries[:16], batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
    out = {"model": MODEL, "max_seq_length": 256, "n_queries": len(queries), "prefix": PREFIX}
    t1 = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        model.encode([q], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
        t1.append(time.perf_counter() - t0)
    t1 = np.array(t1)
    out["batch1_median_ms"] = float(np.median(t1) * 1000)
    out["batch1_mean_ms"] = float(t1.mean() * 1000)
    out["batch1_p95_ms"] = float(np.percentile(t1, 95) * 1000)
    out["batch1_qps"] = float(1.0 / np.median(t1))
    for bs in [16, 32, 64, 128, 200]:
        reps = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.encode(queries, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)
            reps.append(time.perf_counter() - t0)
        out[f"batch{bs}_median_ms"] = float(np.median(reps) * 1000)
        out[f"batch{bs}_qps"] = float(len(queries) / np.median(reps))
    with open(os.path.join(WORK, "query_latency.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
