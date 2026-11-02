import json, os, re, statistics, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/daniel/tnc")
from gateway import MemoryGateway

WORK = "/home/daniel/tnc/work"
N = 10485760
SYSTEM = ("You are a memory-aware assistant. Resolve the user's project reference from the conversation. "
          "If a project is clear, call search_memory with the canonical project name and a self-contained "
          "query. If no project is clear, call ask_clarification. For small talk, call no_tool. "
          "Reply with JSON only.")

def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else None

def main():
    meta = json.load(open(os.path.join(WORK, f"needles_{N}.json"), encoding="utf-8"))
    facts = meta["facts"]
    docs = []
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    with open(os.path.join(WORK, f"mod_texts_{N}.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            docs[o["pos"]] = o["text"]

    t0 = time.time()
    gw = MemoryGateway(docs)
    build_s = time.time() - t0
    print("build", round(build_s, 1), flush=True)

    cases = []
    for f in facts[:100]:
        cases.append(("pronoun", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"We're starting project {f['name']}."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"What's the {f['noun']} for it?"},
        ], f))
    for f in facts[100:150]:
        cases.append(("explicit", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"what is the {f['noun']} for project {f['name']}?"},
        ], f))
    for f in facts[150:200]:
        n = f["name"]
        typo = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        cases.append(("typo", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"We're starting project {n}."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"What's the {f['noun']} for {typo}?"},
        ], f))

    for c in cases[:3]:
        gw.handle(c[1])

    by = {k: {"n": 0, "search": 0, "project": 0, "answer": 0} for k in ("pronoun", "explicit", "typo")}
    timings = {"tool": [], "search": [], "rerank": [], "reader": [], "total": []}
    examples = []
    for kind, msgs, f in cases:
        res = gw.handle(msgs)
        b = by[kind]; b["n"] += 1
        pred = res.get("tool")
        if isinstance(pred, dict) and pred.get("tool") == "search_memory":
            b["search"] += 1
            if pred.get("args", {}).get("project") == f["name"]:
                b["project"] += 1
        if res.get("answer") == f["code"]:
            b["answer"] += 1
        for k in timings:
            if k in res.get("timings", {}):
                timings[k].append(res["timings"][k] * 1000)
        if len(examples) < 8:
            examples.append({"kind": kind, "project": f["name"], "gold": f["code"],
                             "pred": pred, "answer": res.get("answer")})
    n = len(cases)
    out = {
        "N": N, "n": n, "build_s": round(build_s, 1),
        "gpu_max_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "rss_gb": round(int(re.search(r"VmRSS:\s+(\d+)", open("/proc/self/status").read()).group(1)) / 1e6, 2),
        "by_case": {k: {**v, "search_rate": v["search"] / max(1, v["n"]),
                        "project_rate": v["project"] / max(1, v["n"]),
                        "answer_rate": v["answer"] / max(1, v["n"])} for k, v in by.items()},
        "overall_answer": sum(v["answer"] for v in by.values()) / n,
        "latency_ms": {k: {"median": round(statistics.median(v), 1) if v else None,
                           "p95": round(pct(v, 95), 1) if v else None,
                           "n": len(v)} for k, v in timings.items()},
        "examples": examples,
    }
    json.dump(out, open(os.path.join(WORK, "gateway_975m_metrics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
