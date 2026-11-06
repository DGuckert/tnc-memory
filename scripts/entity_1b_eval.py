import json, os, re, time
import numpy as np

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 1_000_000_000
CHUNK = 128
n_chunks = N // CHUNK

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

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
    src = json.load(open(os.path.join(WORK, "needles_10485760.json"), encoding="utf-8"))
    facts = src["facts"]
    rng = np.random.default_rng(20261030 + N)
    pos = rng.choice(n_chunks, size=800, replace=False)
    gold = pos[:200]
    dist = pos[200:].reshape(200, 3)
    mods = {}
    fact_map = {}
    entities = set()
    for i, f in enumerate(facts):
        gp = int(gold[i])
        text = f"Note to self: the {f['noun']} for project {f['name']} is {f['code']}."
        mods[gp] = text
        fact_map[(f["name"].lower(), f["noun"].lower())] = f["code"]
        entities.add(f["name"].lower())
        for j in range(3):
            dp = int(dist[i, j])
            dname, dcode = f["distractors"][j]
            text = f"Note to self: the {f['noun']} for project {dname} is {dcode}."
            mods[dp] = text
            entities.add(dname.lower())
    entities = sorted(entities)
    pattern = re.compile(r"the (.+?) for project ([A-Za-z0-9]+) is ([A-Z0-9]{3}-[A-Z0-9]{4})")

    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        base_docs = [json.loads(line)["text"] for line in f]
    postings = {}
    t0 = time.time()
    hits = 0
    for i in range(n_chunks):
        text = mods.get(i)
        if text is None:
            text = base_docs[i % len(base_docs)]
        if "Note to self:" not in text:
            continue
        m = pattern.search(text)
        if m:
            postings.setdefault(m.group(2).lower(), []).append(i)
            hits += 1
    build_s = time.time() - t0
    print("entities", len(entities), "postings", len(postings), "hits", hits,
          "build_s", round(build_s, 1), flush=True)

    exact_ok = 0
    typo_ok = 0
    fact_ok = 0
    typo_examples = []
    for i, f in enumerate(facts):
        name = f["name"].lower()
        p = postings.get(name, [])
        if p and p[0] == int(gold[i]):
            exact_ok += 1
        n = f["name"]
        typo = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        cands = sorted((osa(typo.lower(), e), e) for e in entities)[:3]
        corrected = cands[0][1] if cands and cands[0][0] <= 2 else None
        if corrected:
            p = postings.get(corrected, [])
            if p and p[0] == int(gold[i]):
                typo_ok += 1
        if fact_map.get((f["name"].lower(), f["noun"].lower())) == f["code"]:
            fact_ok += 1
        if len(typo_examples) < 10:
            typo_examples.append({"query_name": n, "typo": typo, "corrected": corrected,
                                  "gold_pos": int(gold[i]), "post_pos": int(p[0]) if p else None})
    out = {
        "N": N, "n_chunks": n_chunks, "content_tokens": n_chunks * CHUNK,
        "entities": len(entities), "postings": len(postings),
        "build_s": round(build_s, 1),
        "exact_recall@1": exact_ok / len(facts),
        "typo_entity_recall@1": typo_ok / len(facts),
        "structured_fact_acc": fact_ok / len(facts),
        "typo_examples": typo_examples,
        "note": "1B logical-token entity-index test; 1B store is repeated filler with 800 facts injected once."
    }
    json.dump(out, open(os.path.join(WORK, "entity_1b_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "typo_examples"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
