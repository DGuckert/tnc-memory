import json, os, re
import numpy as np
BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 1_000_000_000
CHUNK = 128
n_chunks = N // CHUNK
PURPOSES = [
    "load the model shard","clear the cuda cache","retry the failed download","normalize the token ids",
    "split the long context","cache the embedding batch","validate the config keys","stream the logits",
    "pack the kv cache","quantize the weights","profile the gpu memory","route the api request",
    "merge the lora adapters","resolve the import path","repair the broken checkpoint","pin the dependency versions",
    "dry-run the install command","collect the traceback context","warm up the inference server","dump the attention mask",
]
def main():
    src = json.load(open(os.path.join(WORK, "needles_10485760.json"), encoding="utf-8"))
    facts = src["facts"]
    rng = np.random.default_rng(20261030 + N)
    pos = rng.choice(n_chunks, size=800, replace=False)
    gold = pos[:200]
    mods = {}
    for i in range(800):
        p_idx = i % len(PURPOSES)
        mods[int(pos[i])] = p_idx
    purpose_postings = {}
    for p, idx in mods.items():
        purpose_postings.setdefault(idx, []).append(p)
    r1 = r10 = r50 = 0
    for i in range(200):
        p_idx = i % len(PURPOSES)
        postings = sorted(purpose_postings[p_idx])
        rank = postings.index(int(gold[i])) + 1 if int(gold[i]) in postings else 0
        r1 += rank == 1
        r10 += 1 <= rank <= 10
        r50 += 1 <= rank <= 50
    out = {
        "N": N, "n_chunks": n_chunks, "content_tokens": n_chunks * CHUNK,
        "purpose_only_recall@1": r1 / 200,
        "purpose_only_recall@10": r10 / 200,
        "purpose_only_recall@50": r50 / 200,
        "candidates_per_purpose": len(purpose_postings[0]),
        "note": "1B coding description-only test: no unique project/function name, only the task description."
    }
    json.dump(out, open(os.path.join(WORK, "coding_1b_no_name_results.json"), "w"), indent=1)
    print(json.dumps(out, indent=2), flush=True)
if __name__ == "__main__":
    main()
