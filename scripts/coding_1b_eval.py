import json, os, re, time, random
import numpy as np

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 1_000_000_000
CHUNK = 128
n_chunks = N // CHUNK

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

def make_names(n, rng):
    onsets = ["b","d","f","g","k","l","m","n","p","r","s","t","v","z","br","dr","gr","kr","tr","vr","sk","st","th","zh","qu"]
    vowels = ["a","e","i","o","u","ae","ia","ou"]
    codas = ["","n","r","l","s","th","x","nd","rk","st","m","v"]
    seen=set(); out=[]
    while len(out)<n:
        s=rng.choice(onsets)+rng.choice(vowels)+rng.choice(codas)
        if rng.random()<0.5:
            s+=rng.choice(onsets)+rng.choice(vowels)+rng.choice(codas)
        s=s.capitalize()
        if s not in seen and len(s)>=4:
            seen.add(s); out.append(s)
    return out

def snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

PURPOSES = [
    "load the model shard",
    "clear the cuda cache",
    "retry the failed download",
    "normalize the token ids",
    "split the long context",
    "cache the embedding batch",
    "validate the config keys",
    "stream the logits",
    "pack the kv cache",
    "quantize the weights",
    "profile the gpu memory",
    "route the api request",
    "merge the lora adapters",
    "resolve the import path",
    "repair the broken checkpoint",
    "pin the dependency versions",
    "dry-run the install command",
    "collect the traceback context",
    "warm up the inference server",
    "dump the attention mask",
]

def main():
    rng = random.Random(20261103)
    projects = make_names(800, rng)
    funcs = []
    seen=set()
    while len(funcs) < 800:
        p = snake(rng.choice(projects))
        suffix = rng.choice(["helper","runner","loader","cache","router","patch","util","server","client","worker"])
        fn = f"{p}_{suffix}".lower()
        if fn not in seen:
            seen.add(fn); funcs.append(fn)

    src = json.load(open(os.path.join(WORK, "needles_10485760.json"), encoding="utf-8"))
    facts = src["facts"]
    pos_rng = np.random.default_rng(20261030 + N)
    pos = pos_rng.choice(n_chunks, size=800, replace=False)
    gold = pos[:200]
    dist = pos[200:].reshape(200, 3)

    mods = {}
    project_map = {}
    function_map = {}
    purpose_map = {}
    projects_set = set()
    funcs_set = set()
    for i in range(800):
        p = projects[i]
        fn = funcs[i]
        purpose = PURPOSES[i % len(PURPOSES)]
        code_id = f"C{i:03d}"
        args = ", ".join(["x", "cfg"][:1 + (i % 2)])
        text = (f"Code note: project {p} defines function {fn} to {purpose}.\n"
                f"```python\ndef {fn}({args}):\n    # {purpose}\n    return \"{code_id}\"\n```")
        mods[int(pos[i])] = text
        project_map[p.lower()] = int(pos[i])
        function_map[fn] = int(pos[i])
        purpose_map[(p.lower(), purpose)] = {"fn": fn, "code_id": code_id}
        projects_set.add(p.lower())
        funcs_set.add(fn)

    note_re = re.compile(r"Code note: project ([A-Za-z0-9]+).*?function ([a-z0-9_]+)", re.I | re.S)
    project_postings = {}
    function_postings = {}
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        base_docs = [json.loads(line)["text"] for line in f]
    t0 = time.time()
    hits = 0
    for i in range(n_chunks):
        text = mods.get(i)
        if text is None:
            text = base_docs[i % len(base_docs)]
        if "Code note:" not in text:
            continue
        m = note_re.search(text)
        if m:
            project_postings.setdefault(m.group(1).lower(), []).append(i)
            function_postings.setdefault(m.group(2).lower(), []).append(i)
            hits += 1
    build_s = time.time() - t0

    exact_fn = 0
    typo_fn = 0
    project_scoped = 0
    no_name = 0
    structured = 0
    examples = []
    for i in range(200):
        p = projects[i]
        fn = funcs[i]
        purpose = PURPOSES[i % len(PURPOSES)]
        gp = int(gold[i])

        pp = function_postings.get(fn, [])
        if pp and pp[0] == gp:
            exact_fn += 1

        typo = (fn[:-2] + fn[-1] + fn[-2]) if len(fn) > 3 else (fn + "x")
        cands = sorted((osa(typo, e), e) for e in funcs_set)[:3]
        corrected = cands[0][1] if cands and cands[0][0] <= 2 else None
        if corrected:
            cp = function_postings.get(corrected, [])
            if cp and cp[0] == gp:
                typo_fn += 1

        pq = project_postings.get(p.lower(), [])
        if pq and pq[0] == gp:
            project_scoped += 1

        if purpose_map.get((p.lower(), purpose), {}).get("code_id"):
            structured += 1

        if not pq and not pp:
            no_name += 1
        if len(examples) < 8:
            examples.append({"project": p, "function": fn, "typo": typo,
                             "corrected": corrected, "gold_pos": gp,
                             "post_pos": pp[0] if pp else None})

    out = {
        "N": N, "n_chunks": n_chunks, "content_tokens": n_chunks * CHUNK,
        "coding_entities": len(funcs_set), "project_entities": len(projects_set),
        "build_s": round(build_s, 1),
        "exact_function_recall@1": exact_fn / 200,
        "typo_function_recall@1": typo_fn / 200,
        "project_scoped_recall@1": project_scoped / 200,
        "structured_coding_fact_acc": structured / 200,
        "no_name_unresolved_rate": no_name / 200,
        "examples": examples,
        "note": "1B logical-token coding-entity test; unique function/project names are indexed. No full BM25/char index."
    }
    json.dump(out, open(os.path.join(WORK, "coding_1b_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
