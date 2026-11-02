import json, os
import numpy as np

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
CHUNK = 128
N = 100_000_000
n_chunks = N // CHUNK

def main():
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        base = [json.loads(line) for line in f]
    print("base chunks", len(base), "target", n_chunks, flush=True)
    src = json.load(open(os.path.join(WORK, "needles_10485760.json"), encoding="utf-8"))
    facts = src["facts"]
    assert len(facts) == 200

    srng = np.random.default_rng(20261030 + N)
    pos = srng.choice(n_chunks, size=800, replace=False)
    gold_pos = pos[:200]
    dist_pos = pos[200:].reshape(200, 3)

    mod_positions = []
    mod_texts = []
    for i, f in enumerate(facts):
        gp = int(gold_pos[i])
        base_text = base[gp % len(base)]["text"]
        mod_positions.append(gp)
        mod_texts.append(base_text + "\n" + f"Note to self: the {f['noun']} for project {f['name']} is {f['code']}.")
        for j in range(3):
            dp = int(dist_pos[i, j])
            dname, dcode = f["distractors"][j]
            base_text = base[dp % len(base)]["text"]
            mod_positions.append(dp)
            mod_texts.append(base_text + "\n" + f"Note to self: the {f['noun']} for project {dname} is {dcode}.")

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
        "query_prefix": src["query_prefix"],
        "fact_template": src["fact_template"],
        "fact_token_mean": src["fact_token_mean"],
        "fact_token_max": src["fact_token_max"],
        "fact_tokens_total_injected": src["fact_tokens_total_injected"],
        "note": "100M-token scale test: 9.75M corpus repeated to 781,250 chunks; 800 facts injected once.",
    }
    with open(os.path.join(WORK, "needles_100000000.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    with open(os.path.join(WORK, "mod_texts_100000000.jsonl"), "w", encoding="utf-8") as fh:
        for p, txt in zip(mod_positions, mod_texts):
            fh.write(json.dumps({"pos": int(p), "text": txt}, ensure_ascii=False) + "\n")
    print("wrote", len(mod_positions), "modified chunks", flush=True)

if __name__ == "__main__":
    main()
