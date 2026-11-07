"""Vague coding-query eval over the code index (no function or file names in the questions).

Hit = the target symbol, or a symbol with the same qualname in the same repo (copy-pasted helpers,
e.g. jarvis_bot.py and jarvis_selfbot.py share functions).
Conditions: bm25 / dense / hybrid / hybrid+rerank, each global (all 9 repos incl. two large
third-party clones) and repo-scoped (repo known from the conversation, as the tracker would give).
"""
import json, statistics, sys, time
import numpy as np
sys.path.insert(0, "/home/daniel/tnc")
from code_index import CodeIndex
from code_eval_candidates import REPOS

CAND = "/home/daniel/tnc/work/code_eval/candidates.jsonl"
QS = "/home/daniel/tnc/work/code_eval/code_q.txt"
OUT = "/home/daniel/tnc/work/code_eval/code_eval_metrics.json"


def main():
    cands = [json.loads(l) for l in open(CAND)]
    rows = []
    for line in open(QS):
        if line.strip():
            n, q = line.strip().split("|", 1)
            c = cands[int(n)]
            rows.append({"n": int(n), "q": q, "repo": c["repo"], "qualname": c["qualname"], "idx": c["idx"]})
    t0 = time.time()
    ci = CodeIndex(REPOS, rerank=True)
    build = time.time() - t0
    res = {"n": len(rows), "index": ci.stats(), "build_s": round(build, 1), "conditions": {}}
    for mode, rr in (("bm25", False), ("dense", False), ("hybrid", False), ("hybrid", True)):
        for scoped in (False, True):
            ranks, lat = [], []
            for r in rows:
                t = time.perf_counter()
                hits = ci.search(r["q"], k=50, repo=r["repo"] if scoped else None,
                                 kinds=("function", "method", "class"), mode=mode, rerank=rr)
                lat.append((time.perf_counter() - t) * 1000)
                rank = None
                for j, h in enumerate(hits):
                    if h["ref"]["repo"] == r["repo"] and h["ref"]["name"] == r["qualname"]:
                        rank = j + 1
                        break
                ranks.append(rank)
            n = len(rows)
            key = f"{mode}{'+rerank' if rr else ''}|{'repo_scoped' if scoped else 'global'}"
            res["conditions"][key] = {f"r@{k}": round(sum(1 for x in ranks if x and x <= k) / n, 3) for k in (1, 5, 10, 50)}
            res["conditions"][key]["mrr"] = round(sum(1 / x for x in ranks if x) / n, 3)
            res["conditions"][key]["latency_ms_median"] = round(statistics.median(lat), 1)
            print(key, json.dumps(res["conditions"][key]), flush=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
