import argparse, json, os, random, re
from collections import defaultdict

NOUNS = ["access code", "invoice number", "serial number", "passphrase", "license key", "tracking ID"]
SEED = 20261102
N_TRAIN_PROJECTS = 1200
N_VAL_PROJECTS = 180
N_TEST_PROJECTS = 180

TOOL_SYSTEM = (
    "You are a memory-aware assistant. Resolve the user's project reference from the conversation. "
    "If a project is clear, call search_memory with the canonical project name and a self-contained "
    "query. If no project is clear, call ask_clarification. For small talk, call no_tool. "
    "Reply with JSON only."
)
READER_SYSTEM = (
    "You answer only from the memory notes. If the answer is not present, say exactly: I don't know."
)

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

def load_benchmark():
    path = "/home/daniel/tnc/work/needles_10485760.json"
    data = json.load(open(path, encoding="utf-8"))
    names = {f["name"] for f in data["facts"]}
    codes = {f["code"] for f in data["facts"]}
    for f in data["facts"]:
        for n, c in f["distractors"]:
            names.add(n)
            codes.add(c)
    return names, codes

def fact(p):
    return f"Note to self: the {p['noun']} for project {p['name']} is {p['code']}."

def make_projects(n, rng, start, names_pool, codes_pool):
    out = []
    for i in range(n):
        p = {
            "id": f"p{start + i:04d}",
            "name": names_pool[start + i],
            "code": codes_pool[start + i],
            "noun": NOUNS[(start + i) % len(NOUNS)],
        }
        out.append(p)
    return out

def by_noun(projects):
    d = defaultdict(list)
    for p in projects:
        d[p["noun"]].append(p)
    return d

INTROS = [
    "We're starting a new project called {name}.",
    "Project {name} is now active.",
    "I'm working on {name} today.",
    "The client for {name} just emailed us.",
    "Set up the workspace for {name}.",
    "We need to finish {name} by Friday.",
    "Please add {name} to the project list.",
]
ACKS = ["Got it.", "Noted.", "Okay.", "Understood.", "Thanks for the update."]
SMALL_TALK = ["Hey, how's it going?", "Thanks for the help.", "Can we talk later?", "Good morning."]

def question(noun, ref, rng, name=None):
    if ref == "explicit":
        forms = [
            f"What's the {noun} for project {name}?",
            f"Can you look up the {noun} for {name}?",
            f"Do you have the {noun} for project {name}?",
            f"What is the {noun} for {name}?",
        ]
    elif ref == "pronoun":
        forms = [
            f"What's the {noun} for it?",
            f"Do you have its {noun}?",
            f"What's the {noun} for that project?",
            f"Can you get the {noun} for it?",
        ]
    elif ref == "other":
        forms = [f"And the {noun} for the other one?", f"What about the {noun} for the other project?"]
    elif ref == "new":
        forms = [f"What's the {noun} for the new one?", f"And the {noun} for the new project?"]
    elif ref == "old":
        forms = [f"What's the {noun} for the old one?", f"And the {noun} for the older project?"]
    elif ref == "first":
        forms = [f"What's the {noun} for the first one?", f"And the {noun} for the first project?"]
    elif ref == "second":
        forms = [f"What's the {noun} for the second one?", f"And the {noun} for the second project?"]
    else:
        forms = [f"What's the {noun}?"]
    return rng.choice(forms)

def typo_name(name, rng, all_names):
    variants = [
        name[:-2] + name[-1] + name[-2],
        name[:-1],
        name + name[-1],
        name[0] + name[2:],
    ]
    rng.shuffle(variants)
    for v in variants:
        if v != name and v not in all_names and len(v) >= 4:
            return v
    return name

def tool_target(project, rng):
    return {
        "tool": "search_memory",
        "args": {
            "project": project["name"],
            "query": f"what is the {project['noun']} for project {project['name']}?",
        },
    }

def build_tool_example(projects, bnoun, rng, idx, all_names):
    r = rng.random()
    if r < 0.05:
        user = rng.choice(SMALL_TALK)
        target = {"tool": "no_tool", "args": {}}
        messages = [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ]
        return {
            "id": f"tool-train-{idx:06d}", "task": "tool_call",
            "messages": messages, "target": target,
            "meta": {"case": "small_talk", "reference": "none", "typo": False, "n_context": 0},
        }
    if r < 0.15:
        p = rng.choice(projects)
        user = question(p["noun"], "pronoun", rng)
        target = {"tool": "ask_clarification", "args": {"question": "Which project do you mean?"}}
        messages = [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ]
        return {
            "id": f"tool-train-{idx:06d}", "task": "tool_call",
            "messages": messages, "target": target,
            "meta": {"case": "clarify", "reference": "pronoun", "typo": False, "n_context": 0},
        }

    case = rng.choices(
        ["explicit", "pronoun", "other", "new", "old", "first", "second", "unmentioned"],
        weights=[30, 26, 8, 8, 6, 6, 6, 10], k=1
    )[0]
    if case in ("other", "first", "second"):
        n_ctx = 2
    elif case == "unmentioned":
        n_ctx = rng.choice([1, 2])
    else:
        n_ctx = rng.choice([1, 2, 3])

    if case == "unmentioned":
        target = rng.choice(projects)
        ctx = rng.sample([p for p in projects if p["id"] != target["id"]], n_ctx)
        order = ctx
    else:
        target = rng.choice(projects)
        pool = [p for p in projects if p["id"] != target["id"]]
        others = rng.sample(pool, n_ctx - 1)
        if case == "pronoun":
            order = others + [target]
        elif case in ("new", "second"):
            order = others + [target]
        elif case in ("other", "old", "first"):
            order = [target] + others
        else:
            order = others + [target]
            rng.shuffle(order)

    turns = [{"role": "system", "content": TOOL_SYSTEM}]
    for p in order:
        turns.append({"role": "user", "content": rng.choice(INTROS).format(name=p["name"])})
        turns.append({"role": "assistant", "content": rng.choice(ACKS)})
    if rng.random() < 0.3:
        turns.append({"role": "user", "content": rng.choice(["Thanks.", "Okay.", "Makes sense."])})
        turns.append({"role": "assistant", "content": "No problem."})

    typo = False
    qname = target["name"]
    ref = case
    if case in ("explicit", "unmentioned") and rng.random() < 0.25:
        qname = typo_name(target["name"], rng, all_names)
        typo = qname != target["name"]
    if case == "unmentioned":
        ref = "explicit"
        qname = target["name"]
    user_q = question(target["noun"], ref, rng, name=qname)
    target_json = tool_target(target, rng)
    turns.append({"role": "user", "content": user_q})
    turns.append({"role": "assistant", "content": json.dumps(target_json, separators=(",", ":"))})
    return {
        "id": f"tool-train-{idx:06d}", "task": "tool_call",
        "messages": turns, "target": target_json,
        "meta": {"case": case, "reference": ref, "typo": typo, "n_context": len(order),
                 "target_project": target["name"], "target_noun": target["noun"]},
    }

def build_reader_example(projects, bnoun, rng, idx):
    gold = rng.choice(projects)
    present = rng.random() < 0.85
    same = [p for p in bnoun[gold["noun"]] if p["id"] != gold["id"]]
    selected_ids = {gold["id"]}
    if present:
        distractors = rng.sample(same, 3)
        notes = [fact(gold)] + [fact(p) for p in distractors]
        selected_ids.update(p["id"] for p in distractors)
        target = gold["code"]
    else:
        absent = rng.sample(same, 4)
        notes = [fact(p) for p in absent]
        selected_ids.update(p["id"] for p in absent)
        target = "I don't know."
    # add random facts, never a project already in the notes
    pool = [p for p in projects if p["id"] not in selected_ids]
    extra = rng.sample(pool, rng.choice([0, 1, 2]))
    notes += [fact(p) for p in extra]
    rng.shuffle(notes)
    user = "Memory notes:\n" + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
    user += f"\nQuestion: what is the {gold['noun']} for project {gold['name']}?"
    messages = [
        {"role": "system", "content": READER_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]
    return {
        "id": f"reader-train-{idx:06d}", "task": "answer_from_memory",
        "messages": messages, "target": target,
        "meta": {"gold_present": present, "gold_project": gold["name"],
                 "gold_noun": gold["noun"], "n_notes": len(notes)},
    }

def build_rerank_example(projects, bnoun, rng, idx):
    gold = rng.choice(projects)
    same = [p for p in bnoun[gold["noun"]] if p["id"] != gold["id"]]
    negs = [fact(p) for p in rng.sample(same, 3)]
    other_noun = rng.choice([n for n in NOUNS if n != gold["noun"]])
    other_code = rng.choice([p["code"] for p in projects if p["code"] != gold["code"]])
    negs.append(f"Note to self: the {other_noun} for project {gold['name']} is {other_code}.")
    negs.append(fact(rng.choice([p for p in projects if p["id"] != gold["id"]])))
    seen = set()
    uniq = []
    for n in negs:
        if n != fact(gold) and n not in seen:
            seen.add(n)
            uniq.append(n)
    return {
        "id": f"rerank-train-{idx:06d}",
        "query": f"what is the {gold['noun']} for project {gold['name']}?",
        "positive": fact(gold),
        "negatives": uniq,
        "meta": {"gold_project": gold["name"], "gold_noun": gold["noun"]},
    }

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/daniel/tnc-repo/training")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bench_names, bench_codes = load_benchmark()
    rng = random.Random(SEED)
    all_names = []
    while len(all_names) < N_TRAIN_PROJECTS + N_VAL_PROJECTS + N_TEST_PROJECTS:
        for n in make_names(500, rng):
            if n not in bench_names and n not in all_names:
                all_names.append(n)
    all_codes = make_codes(N_TRAIN_PROJECTS + N_VAL_PROJECTS + N_TEST_PROJECTS, rng)
    all_codes = [c for c in all_codes if c not in bench_codes]
    while len(all_codes) < len(all_names):
        all_codes += [c for c in make_codes(200, rng) if c not in bench_codes and c not in all_codes]

    total = N_TRAIN_PROJECTS + N_VAL_PROJECTS + N_TEST_PROJECTS
    names = all_names[:total]
    codes = all_codes[:total]
    train = make_projects(N_TRAIN_PROJECTS, rng, 0, names, codes)
    val = make_projects(N_VAL_PROJECTS, rng, N_TRAIN_PROJECTS, names, codes)
    test = make_projects(N_TEST_PROJECTS, rng, N_TRAIN_PROJECTS + N_VAL_PROJECTS, names, codes)
    assert not ({p["name"] for p in train} & {p["name"] for p in val})
    assert not ({p["name"] for p in train} & {p["name"] for p in test})
    assert not ({p["name"] for p in val} & {p["name"] for p in test})
    all_name_set = set(names) | bench_names

    def make_split(projects, split):
        bnoun = by_noun(projects)
        tool_n = 5000 if split == "train" else 750
        reader_n = 5000 if split == "train" else 750
        rerank_n = 10000 if split == "train" else 1500
        rng_t = random.Random(SEED + {"train": 1, "val": 2, "test": 3}[split])
        rng_r = random.Random(SEED + {"train": 4, "val": 5, "test": 6}[split])
        rng_k = random.Random(SEED + {"train": 7, "val": 8, "test": 9}[split])
        tool = [build_tool_example(projects, bnoun, rng_t, i, all_name_set) for i in range(1, tool_n + 1)]
        reader = [build_reader_example(projects, bnoun, rng_r, i) for i in range(1, reader_n + 1)]
        rerank = [build_rerank_example(projects, bnoun, rng_k, i) for i in range(1, rerank_n + 1)]
        for r in tool + reader:
            r["id"] = r["id"].replace("train", split)
        for r in rerank:
            r["id"] = r["id"].replace("train", split)
        return tool, reader, rerank

    train_tool, train_reader, train_rerank = make_split(train, "train")
    val_tool, val_reader, val_rerank = make_split(val, "val")
    test_tool, test_reader, test_rerank = make_split(test, "test")

    write_jsonl(os.path.join(args.out, "train_sft.jsonl"), train_tool + train_reader)
    write_jsonl(os.path.join(args.out, "val_sft.jsonl"), val_tool + val_reader)
    write_jsonl(os.path.join(args.out, "test_sft.jsonl"), test_tool + test_reader)
    write_jsonl(os.path.join(args.out, "train_reranker.jsonl"), train_rerank)
    write_jsonl(os.path.join(args.out, "val_reranker.jsonl"), val_rerank)
    write_jsonl(os.path.join(args.out, "test_reranker.jsonl"), test_rerank)

    def count_task(rows):
        d = defaultdict(int)
        for r in rows:
            d[r["task"]] += 1
        return dict(d)

    manifest = {
        "version": 1,
        "seed": SEED,
        "benchmark_excluded": True,
        "benchmark_names": sorted(bench_names),
        "benchmark_codes": sorted(bench_codes),
        "splits": {
            "train": {"projects": len(train), "sft": len(train_tool) + len(train_reader),
                      "tool": len(train_tool), "reader": len(train_reader),
                      "reranker": len(train_rerank)},
            "val": {"projects": len(val), "sft": len(val_tool) + len(val_reader),
                    "tool": len(val_tool), "reader": len(val_reader),
                    "reranker": len(val_rerank)},
            "test": {"projects": len(test), "sft": len(test_tool) + len(test_reader),
                     "tool": len(test_tool), "reader": len(test_reader),
                     "reranker": len(test_rerank)},
        },
        "tasks": {
            "tool_call": "multi-turn conversation -> JSON tool call (search_memory / ask_clarification / no_tool)",
            "answer_from_memory": "memory notes + question -> exact code or I don't know.",
        },
        "files": [
            "train_sft.jsonl", "val_sft.jsonl", "test_sft.jsonl",
            "train_reranker.jsonl", "val_reranker.jsonl", "test_reranker.jsonl",
        ],
        "format": {
            "sft": "one JSON object per line: id, task, messages, target, meta",
            "reranker": "one JSON object per line: id, query, positive, negatives, meta",
        },
        "notes": "Synthetic project names/codes generated for this dataset; the 200 benchmark facts are excluded.",
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # validation
    def read(path):
        return [json.loads(line) for line in open(path, encoding="utf-8")]
    all_sft = read(os.path.join(args.out, "train_sft.jsonl")) + read(os.path.join(args.out, "val_sft.jsonl")) + read(os.path.join(args.out, "test_sft.jsonl"))
    ids = [r["id"] for r in all_sft]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in all_sft:
        assert r["messages"][-1]["role"] == "assistant"
        assert r["messages"][-1]["content"] == (r["target"] if isinstance(r["target"], str) else json.dumps(r["target"], separators=(",", ":")))
        if r["task"] == "tool_call":
            assert r["target"]["tool"] in ("search_memory", "ask_clarification", "no_tool")
            if r["target"]["tool"] == "search_memory":
                assert r["target"]["args"]["project"] in all_name_set
        if r["task"] == "answer_from_memory":
            assert r["target"] == "I don't know." or re.fullmatch(r"[A-Z0-9]{3}-[A-Z0-9]{4}", r["target"])
    print(json.dumps(manifest["splits"], indent=2))
    print("wrote", args.out)

if __name__ == "__main__":
    main()
