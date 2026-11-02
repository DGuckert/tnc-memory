import json, os, time, random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
MODEL = "BAAI/bge-small-en-v1.5"
CHUNK = 128
SIZES = [131072, 1048576, 10485760]
SEED_FACTS = 424242
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NOUNS = ["access code", "invoice number", "serial number", "passphrase", "license key", "tracking ID"]

def make_names(n, rng):
    onsets = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
              "br", "dr", "gr", "kr", "tr", "vr", "sk", "st", "th", "zh", "qu"]
    vowels = ["a", "e", "i", "o", "u", "ae", "ia", "ou"]
    codas = ["", "n", "r", "l", "s", "th", "x", "nd", "rk", "st", "m", "v"]
    seen = set()
    out = []
    while len(out) < n:
        s = rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas)
        if rng.random() < 0.5:
            s += rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas)
        s = s.capitalize()
        if s not in seen and len(s) >= 4:
            seen.add(s)
            out.append(s)
    return out

def make_codes(n, rng):
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    seen = set()
    out = []
    while len(out) < n:
        s = "".join(rng.choice(alphabet) for _ in range(7))
        s = s[:3] + "-" + s[3:]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def fact(noun, name, code):
    return f"Note to self: the {noun} for project {name} is {code}."

def query(noun, name):
    return f"what is the {noun} for project {name}?"

def main():
    t0 = time.time()
    with open(os.path.join(WORK, "chunks.jsonl"), "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    total_chunks = len(chunks)
    print("chunks", total_chunks, flush=True)

    rng = random.Random(SEED_FACTS)
    names = make_names(800, rng)
    codes = make_codes(800, rng)
    facts = []
    for i in range(200):
        noun = NOUNS[i % len(NOUNS)]
        name = names[i]
        code = codes[i]
        dist = []
        for j in range(3):
            dist.append((names[200 + i * 3 + j], codes[200 + i * 3 + j]))
        facts.append({"i": i, "noun": noun, "name": name, "code": code, "distractors": dist,
                      "query": query(noun, name)})
    assert len(set(names)) == 800 and len(set(codes)) == 800

    tok = Tokenizer.from_pretrained("gpt2")
    tok.no_truncation(); tok.no_padding()
    fact_token_counts = [len(tok.encode(fact(f["noun"], f["name"], f["code"])).ids) for f in facts]
    print("fact tokens mean/max", np.mean(fact_token_counts), max(fact_token_counts), flush=True)

    torch.manual_seed(0)
    model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 256
    print("model loaded", time.time() - t0, flush=True)

    query_texts = [QUERY_PREFIX + f["query"] for f in facts]
    qvec = model.encode(query_texts, batch_size=64, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False, device="cuda").astype(np.float32)
    np.save(os.path.join(WORK, "query_vectors.npy"), qvec)
    print("query vectors", qvec.shape, flush=True)

    meta_all = {}
    for N in SIZES:
        n_chunks = min(N // CHUNK, total_chunks)
        if n_chunks < 1000:
            print("skip", N, n_chunks, flush=True)
            continue
        srng = np.random.default_rng(20261030 + N)
        pos = srng.choice(n_chunks, size=800, replace=False)
        gold_pos = pos[:200]
        dist_pos = pos[200:].reshape(200, 3)

        mod_positions = []
        mod_texts = []
        for i, f in enumerate(facts):
            gp = int(gold_pos[i])
            mod_positions.append(gp)
            mod_texts.append(chunks[gp]["text"] + "\n" + fact(f["noun"], f["name"], f["code"]))
            for j in range(3):
                dp = int(dist_pos[i, j])
                dname, dcode = f["distractors"][j]
                mod_positions.append(dp)
                mod_texts.append(chunks[dp]["text"] + "\n" + fact(f["noun"], dname, dcode))

        t1 = time.time()
        mvec = model.encode(mod_texts, batch_size=128, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False, device="cuda").astype(np.float32)
        print(f"N={N} n_chunks={n_chunks} modified={len(mod_texts)} encode_s={time.time()-t1:.1f}", flush=True)
        np.save(os.path.join(WORK, f"mod_vectors_{N}.npy"), mvec)

        meta = {
            "N_requested": N,
            "n_chunks": n_chunks,
            "content_tokens": n_chunks * CHUNK,
            "n_needles": 200,
            "n_distractors": 600,
            "n_modified_chunks": len(mod_positions),
            "modified_positions": [int(x) for x in mod_positions],
            "gold_positions": [int(x) for x in gold_pos],
            "distractor_positions": [[int(x) for x in row] for row in dist_pos],
            "facts": facts,
            "query_prefix": QUERY_PREFIX,
            "fact_template": "Note to self: the {noun} for project {Name} is {code}.",
            "fact_token_mean": float(np.mean(fact_token_counts)),
            "fact_token_max": int(max(fact_token_counts)),
            "fact_tokens_total_injected": int(sum(fact_token_counts) * 4),
        }
        with open(os.path.join(WORK, f"needles_{N}.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        meta_all[str(N)] = {k: meta[k] for k in ["N_requested", "n_chunks", "content_tokens",
                                                 "n_needles", "n_distractors", "n_modified_chunks",
                                                 "fact_token_mean", "fact_token_max"]}
        # store modified texts separately for failure analysis
        with open(os.path.join(WORK, f"mod_texts_{N}.jsonl"), "w", encoding="utf-8") as fh:
            for p, txt in zip(mod_positions, mod_texts):
                fh.write(json.dumps({"pos": int(p), "text": txt}, ensure_ascii=False) + "\n")

    with open(os.path.join(WORK, "needle_meta.json"), "w") as fh:
        json.dump(meta_all, fh, indent=2)
    print(json.dumps(meta_all, indent=2), flush=True)

if __name__ == "__main__":
    main()
