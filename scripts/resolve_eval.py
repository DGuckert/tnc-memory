"""Reference-resolution eval: old gateway heuristics vs EntityTracker (vs model routers with --models).

Datasets
  hard       results of make_hard_eval.py (100 rows; written by the previous agent, independent of
             the tracker). Search rows are scored on project; clarify rows on asking.
  reference  make_reference_eval.py (400 rows, held-out phrasings, real chatter).

A prediction is correct if it names the gold project, or (gold = clarify) it declines to name one.
"""
import argparse, json, os, re, sys, time, collections
sys.path.insert(0, "/home/daniel/tnc")
from entity_tracker import EntityTracker

WORK = "/home/daniel/tnc/work"
SLOTS = ["access code", "invoice number", "serial number", "passphrase", "license key", "tracking id", "code", "key"]


def word_vocab(min_count=20):
    c = collections.Counter()
    for l in open("/home/daniel/tnc/corpus.jsonl", encoding="utf-8"):
        c.update(re.findall(r"[a-z]+", json.loads(l)["t"].lower()))
    return {w for w, n in c.items() if n >= min_count}


class OldHeuristics:
    """The gateway's pre-tracker logic (find_exact -> find_context -> find_fuzzy), copied verbatim
    so it can run without loading models."""
    def __init__(self, names, docs):
        from gateway import osa, tokenize
        self.osa, self.tokenize = osa, tokenize
        self.entity_map = {n.lower(): n for n in names}
        self.entities = sorted(self.entity_map)
        self.postings = set(t for d in docs for t in tokenize(d))

    def resolve(self, messages):
        tok = self.tokenize
        text = messages[-1].get("content", "")
        for t in tok(text):
            if t in self.entity_map:
                return self.entity_map[t]
        if re.search(r"\b(what|what's|which|code|number|passphrase|key|id|tracking|invoice|serial|license)\b", text, re.I):
            seq = []
            for m in messages[:-1]:
                for t in tok(m.get("content", "")):
                    if t in self.entity_map:
                        p = self.entity_map[t]
                        if not seq or seq[-1] != p:
                            seq.append(p)
            if seq:
                if re.search(r"\b(other|new|old|first|second)\b", text, re.I):
                    if len(set(seq)) == 1:
                        return seq[-1]
                else:
                    return seq[-1]
        for t in tok(text):
            if t in self.postings or len(t) < 4:
                continue
            cands = sorted((self.osa(t, e), e) for e in self.entities)[:3]
            if cands and cands[0][0] <= 2:
                return self.entity_map[cands[0][1]]
        return None


def load(ds):
    if ds == "hard":
        meta = json.load(open(os.path.join(WORK, "hard_eval_meta.json")))
        names = [p["name"] for p in meta["projects"]]
        docs = [f"Note to self: the {p['noun']} for project {p['name']} is {p['code']}." for p in meta["projects"]]
        rows = []
        for r in map(json.loads, open(os.path.join(WORK, "hard_eval.jsonl"))):
            t = r["target"]
            if t["tool"] == "no_tool":
                continue
            gold = t["args"]["project"] if t["tool"] == "search_memory" else None
            rows.append({"id": r["id"], "case": r["meta"]["case"], "messages": r["messages"][1:-1], "gold": gold,
                         "n_projects": None})
        return names, docs, rows
    d = os.path.join(WORK, "reference_eval")
    held = ds == "heldout"
    projects = json.load(open(os.path.join(d, "projects_heldout.json" if held else "projects.json")))["projects"]
    names = [p["name"] for p in projects]
    docs = [f"Note to self: the {nn} for project {p['name']} is {c}." for p in projects for nn, c in p["facts"].items()]
    rows = [{"id": r["id"], "case": r["case"], "messages": r["messages"], "gold": r["target_project"]}
            for r in map(json.loads, open(os.path.join(d, "reference_heldout.jsonl" if held else "reference_eval.jsonl")))]
    return names, docs, rows


def score(pred, gold):
    return pred == gold


def run_models(names, docs, rows, which):
    """Run the 135M tool router / 360M middleman on the same rows (project field of the tool call)."""
    from gateway import load_adapter
    import torch
    path = {"tool135m": "/home/daniel/tnc/training_runs/tool-135m-lora",
            "middleman360m": "/home/daniel/tnc/training_runs/middleman-lora"}[which]
    tok, model = load_adapter(path)
    SYSTEM = ("You are a memory-aware assistant. Resolve the user's project reference from the conversation. "
              "If a project is clear, call search_memory with the canonical project name and a self-contained "
              "query. If no project is clear, call ask_clarification. For small talk, call no_tool. "
              "Reply with JSON only.")
    preds = []
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM}] + r["messages"]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            o = json.loads(text)
            p = o.get("args", {}).get("project") if o.get("tool") == "search_memory" else None
        except Exception:
            p = None
        preds.append(p)
    del model
    torch.cuda.empty_cache()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    a = ap.parse_args()
    vocab = word_vocab()
    out = {}
    for ds in ("hard", "reference", "heldout"):
        names, docs, rows = load(ds)
        old = OldHeuristics(names, docs)
        tr = EntityTracker(names, vocab=vocab, slot_words=SLOTS)
        systems = {"old_heuristics": [old.resolve(r["messages"]) for r in rows]}
        t0 = time.perf_counter()
        tres = [tr.resolve(r["messages"]) for r in rows]
        tr_ms = (time.perf_counter() - t0) * 1000 / len(rows)
        systems["tracker"] = [x["entity"] for x in tres]
        if a.models:
            for m in ("tool135m", "middleman360m"):
                systems[m] = run_models(names, docs, rows, m)
        res = {"n": len(rows), "tracker_ms_per_query": round(tr_ms, 3), "systems": {}}
        for s, preds in systems.items():
            by = collections.defaultdict(lambda: [0, 0])
            for r, p in zip(rows, preds):
                by[r["case"]][0] += 1
                by[r["case"]][1] += score(p, r["gold"])
            res["systems"][s] = {"acc": round(sum(v[1] for v in by.values()) / len(rows), 3),
                                 "by_case": {c: round(v[1] / v[0], 3) for c, v in sorted(by.items())},
                                 "n_by_case": {c: v[0] for c, v in sorted(by.items())}}
        res["tracker_errors"] = [{"id": r["id"], "case": r["case"], "gold": r["gold"], "pred": x["entity"],
                                  "how": x["how"], "cands": x["candidates"], "last": r["messages"][-1]["content"]}
                                 for r, x in zip(rows, tres) if x["entity"] != r["gold"]][:40]
        out[ds] = res
        print(ds, json.dumps({s: v["acc"] for s, v in res["systems"].items()}), flush=True)
        for s, v in res["systems"].items():
            print("  ", s, v["by_case"])
    json.dump(out, open(os.path.join(WORK, "reference_eval", "resolve_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
