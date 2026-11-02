import json, numpy as np
WORK = "/home/daniel/tnc/work"
CHUNK = 128
SIZES = [131072, 1048576, 10485760]
emb = np.load(f"{WORK}/embeddings_fp32.npy")
q = np.load(f"{WORK}/query_vectors.npy")
out = {}
for N in SIZES:
    meta = json.load(open(f"{WORK}/needles_{N}.json"))
    n = meta["n_chunks"]
    store = emb[:n].copy()
    mv = np.load(f"{WORK}/mod_vectors_{N}.npy").astype(np.float32)
    store[np.array(meta["modified_positions"])] = mv
    facts = meta["facts"]
    gold = np.array(meta["gold_positions"])
    # map modified position -> fact index and noun
    pos_noun = {}
    pos_fact = {}
    for i, f in enumerate(facts):
        pos_fact[meta["gold_positions"][i]] = i
        pos_noun[meta["gold_positions"][i]] = f["noun"]
        for j in range(3):
            p = meta["distractor_positions"][i][j]
            pos_fact[p] = i
            pos_noun[p] = f["noun"]
    same_noun = {noun: set(p for p, no in pos_noun.items() if no == noun) for noun in set(pos_noun.values())}
    sims = q @ store.T
    order = np.argsort(-sims, axis=1)
    r = np.array([int(np.where(order[i] == gold[i])[0][0]) + 1 for i in range(len(gold))])
    top1_same = 0; top10_same = 0; top1_any = 0; top10_any = 0
    for i, f in enumerate(facts):
        t1 = int(order[i, 0])
        t10 = set(int(x) for x in order[i, :10])
        sn = same_noun[f["noun"]] - {int(gold[i])}
        if t1 in sn: top1_same += 1
        if t10 & sn: top10_same += 1
        if t1 in pos_fact and t1 != int(gold[i]): top1_any += 1
        if (t10 & set(pos_fact.keys())) - {int(gold[i])}: top10_any += 1
    out[str(N)] = {
        "n_chunks": n,
        "recall@1": float(np.mean(r <= 1)),
        "recall@10": float(np.mean(r <= 10)),
        "recall@50": float(np.mean(r <= 50)),
        "same_noun_top1_rate": top1_same / len(facts),
        "same_noun_top10_rate": top10_same / len(facts),
        "any_fact_top1_rate": top1_any / len(facts),
        "any_fact_top10_rate": top10_any / len(facts),
        "median_rank": int(np.median(r)),
        "n_facts_per_noun": {k: len(v) for k, v in same_noun.items()},
    }
    print(out[str(N)])
json.dump(out, open(f"{WORK}/same_noun_analysis.json", "w"), indent=2)
