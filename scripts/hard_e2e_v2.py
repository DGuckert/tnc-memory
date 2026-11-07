"""Noisy hard set (100 rows) end-to-end through memory_api.retrieve, same scoring as hard_gateway_eval.py.

search rows   correct if retrieve() returns the gold value as its structured answer
clarify rows  correct if retrieve() asks for clarification
small talk    correct if retrieve() does not run a fact lookup (no slot word -> no memory call)
Also reports how many search rows were answered by asking instead of guessing.
"""
import json, os, sys, collections
sys.path.insert(0, "/home/daniel/tnc")
from memory_api import Memory

WORK = "/home/daniel/tnc/work"
meta = json.load(open(os.path.join(WORK, "hard_eval_meta.json")))
code = {p["name"]: p["code"] for p in meta["projects"]}
docs = [f"Note to self: the {p['noun']} for project {p['name']} is {p['code']}." for p in meta["projects"]]
rows = [json.loads(l) for l in open(os.path.join(WORK, "hard_eval.jsonl"))]
mem = Memory(fact_docs=docs)
by = collections.defaultdict(lambda: {"n": 0, "ok": 0, "asked": 0, "wrong": 0})
errors = []
for r in rows:
    t = r["target"]
    msgs = [m for m in r["messages"][1:-1]]
    out = mem.retrieve(msgs, k=5, sources=("fact",))
    case = r["meta"]["case"]
    b = by[case]; b["n"] += 1
    if t["tool"] == "search_memory":
        gold = code[t["args"]["project"]]
        if out["needs_clarification"]:
            b["asked"] += 1
        elif out["structured"] and out["structured"]["value"] == gold:
            b["ok"] += 1
        else:
            b["wrong"] += 1
            errors.append({"id": r["id"], "case": case, "gold_project": t["args"]["project"], "entity": out["entity"],
                           "how": out["resolution"]["how"], "last": msgs[-1]["content"]})
    elif t["tool"] == "ask_clarification":
        b["ok"] += out["needs_clarification"]
    else:
        b["ok"] += out["structured"] is None and not out["needs_clarification"]
n = len(rows)
res = {"n": n, "end_to_end_acc": sum(v["ok"] for v in by.values()) / n,
       "asked_instead_of_answer": sum(v["asked"] for v in by.values()),
       "wrong_answers": sum(v["wrong"] for v in by.values()),
       "by_case": dict(by), "errors": errors,
       "baseline_gateway_v1": {"end_to_end_acc": 0.94, "wrong_answers": 6}}
json.dump(res, open(os.path.join(WORK, "hard_e2e_v2.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
