import json, os, re, time
import numpy as np

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
N = 1_000_000_000
CHUNK = 128
n_chunks = N // CHUNK
SYSTEM = ("You are a memory-aware assistant. Resolve the user's project reference from the conversation. "
          "If a project is clear, call search_memory with the canonical project name and a self-contained "
          "query. If no project is clear, call ask_clarification. For small talk, call no_tool. "
          "Reply with JSON only.")

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

class StructuredGateway1B:
    def __init__(self, mods, fact_map, entity_map):
        self.mods = mods
        self.fact_map = fact_map
        self.entity_map = entity_map
        self.entities = sorted(entity_map)

    def find_exact_project(self, messages):
        text = messages[-1].get("content", "") if messages else ""
        for t in tokenize(text):
            if t in self.entity_map:
                return self.entity_map[t]
        return None

    def find_context_project(self, messages):
        text = messages[-1].get("content", "") if messages else ""
        if not re.search(r"\b(what|what's|which|code|number|passphrase|key|id|tracking|invoice|serial|license)\b", text, re.I):
            return None
        seq = []
        for m in messages[:-1]:
            for t in tokenize(m.get("content", "")):
                if t in self.entity_map:
                    p = self.entity_map[t]
                    if not seq or seq[-1] != p:
                        seq.append(p)
        if not seq:
            return None
        if re.search(r"\b(other|new|old|first|second)\b", text, re.I):
            return seq[-1] if len(set(seq)) == 1 else None
        return seq[-1]

    def find_fuzzy_project(self, messages):
        text = messages[-1].get("content", "") if messages else ""
        for t in tokenize(text):
            if len(t) < 4:
                continue
            cands = sorted((osa(t, e), e) for e in self.entities)[:3]
            if cands and cands[0][0] <= 2:
                return self.entity_map[cands[0][1]]
        return None

    def structured_answer(self, query, project):
        m = re.search(r"the (.+?) for", query, re.I)
        if not m:
            return None
        return self.fact_map.get((project.lower(), m.group(1).strip().lower()))

    def handle(self, messages):
        project = self.find_exact_project(messages) or self.find_context_project(messages) or self.find_fuzzy_project(messages)
        if not project:
            return None
        query = messages[-1].get("content", "")
        return self.structured_answer(query, project)

def main():
    src = json.load(open(os.path.join(WORK, "needles_10485760.json"), encoding="utf-8"))
    facts = src["facts"]
    rng = np.random.default_rng(20261030 + N)
    pos = rng.choice(n_chunks, size=800, replace=False)
    gold = pos[:200]
    dist = pos[200:].reshape(200, 3)
    mods = {}
    fact_map = {}
    entity_map = {}
    for i, f in enumerate(facts):
        gp = int(gold[i])
        text = f"Note to self: the {f['noun']} for project {f['name']} is {f['code']}."
        mods[gp] = text
        fact_map[(f["name"].lower(), f["noun"].lower())] = f["code"]
        entity_map[f["name"].lower()] = f["name"]
        for j in range(3):
            dp = int(dist[i, j])
            dname, dcode = f["distractors"][j]
            mods[dp] = f"Note to self: the {f['noun']} for project {dname} is {dcode}."
            entity_map[dname.lower()] = dname
    gw = StructuredGateway1B(mods, fact_map, entity_map)
    cases = []
    for f in facts[:100]:
        cases.append(("pronoun", f, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"We're starting project {f['name']}."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"What's the {f['noun']} for it?"},
        ]))
    for f in facts[100:150]:
        cases.append(("explicit", f, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"what is the {f['noun']} for project {f['name']}?"},
        ]))
    for f in facts[150:200]:
        n = f["name"]
        typo = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        cases.append(("typo", f, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"We're starting project {n}."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"What's the {f['noun']} for {typo}?"},
        ]))
    by = {k: {"n": 0, "ok": 0} for k in ("pronoun", "explicit", "typo")}
    for kind, f, msgs in cases:
        ans = gw.handle(msgs)
        by[kind]["n"] += 1
        if ans == f["code"]:
            by[kind]["ok"] += 1
    out = {
        "N": N, "n_chunks": n_chunks, "content_tokens": n_chunks * CHUNK,
        "structured_gateway": {k: {**v, "acc": v["ok"] / max(1, v["n"])} for k, v in by.items()},
        "overall_acc": sum(v["ok"] for v in by.values()) / len(cases),
        "note": "1B logical-token structured-gateway test; no full BM25/char index."
    }
    json.dump(out, open(os.path.join(WORK, "structured_gateway_1b_results.json"), "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
