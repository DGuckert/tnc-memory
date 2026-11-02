import json, os, time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
MODEL = "BAAI/bge-small-en-v1.5"

def main():
    t0 = time.time()
    with open(os.path.join(WORK, "chunks.jsonl"), "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    texts = [c["text"] for c in chunks]
    print(f"chunks={len(texts)} load_s={time.time()-t0:.1f}", flush=True)

    torch.manual_seed(0)
    model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 256
    print("model loaded", time.time() - t0, flush=True)
    print("dtype", next(model.parameters()).dtype, flush=True)

    t1 = time.time()
    emb = model.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        device="cuda",
    )
    t2 = time.time()
    print(f"encode_s={t2-t1:.1f} shape={emb.shape} dtype={emb.dtype} "
          f"chunks_per_s={len(texts)/(t2-t1):.2f}", flush=True)
    print("gpu_mem_allocated_gb", round(torch.cuda.max_memory_allocated() / 1024**3, 2), flush=True)

    emb32 = emb.astype(np.float32)
    emb16 = emb.astype(np.float16)
    np.save(os.path.join(WORK, "embeddings_fp32.npy"), emb32)
    np.save(os.path.join(WORK, "embeddings_fp16.npy"), emb16)

    norms = np.linalg.norm(emb32, axis=1)
    meta = {
        "model": MODEL,
        "n_chunks": len(texts),
        "dim": int(emb32.shape[1]),
        "dtype_compute": str(next(model.parameters()).dtype),
        "batch_size": 128,
        "max_seq_length": 256,
        "normalized": True,
        "encode_seconds": t2 - t1,
        "chunks_per_second": len(texts) / (t2 - t1),
        "gpu_max_memory_allocated_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
    }
    with open(os.path.join(WORK, "embed_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)

if __name__ == "__main__":
    main()
