"""Re-run only the reranked condition with each rerank passage type (window / msg / ctx).

Appends results to real_eval_metrics.json under conditions["hybrid+rerank[<passage>]|<scope>"].
"""
import json, statistics, sys, time
import numpy as np
sys.path.insert(0, "/home/daniel/tnc")
import real_eval_run as R
from message_index import MessageIndex

rows = [json.loads(l) for l in open(R.EVAL, encoding="utf-8")]
res = json.load(open(R.OUT))
idx = MessageIndex()
msgs = idx.msgs
cnt = {}
R.ORD = []
for m in msgs:
    R.ORD.append(cnt.get(m["ch"], 0)); cnt[m["ch"]] = R.ORD[-1] + 1
for passage in ("msg", "ctx"):
    for scope in ("global", "scoped", "channel_oracle"):
        ex, win, lat, by_type = [], [], [], {}
        for r in rows:
            kw = {}
            if scope == "scoped":
                if r["type"] == "A": kw["author"] = r["author"]
                if r["type"] == "C": kw["channel"] = r["channel"]
            if scope == "channel_oracle":
                kw["channel"] = r["channel"]
            t = time.perf_counter()
            ret = [i for i, _ in idx.search(r["question"], k=50, mode="hybrid", rerank=True, rerank_passage=passage, **kw)]
            lat.append((time.perf_counter() - t) * 1000)
            e, w = R.hits(ret, r["target_idx"], msgs)
            ex.append(e); win.append(w); by_type.setdefault(r["type"], []).append(e)
        key = f"hybrid+rerank[{passage}]|{scope}"
        n = len(rows)
        res["conditions"][key] = {"exact": R.summarize(ex, n), "window": R.summarize(win, n),
                                  "exact_by_type": {t: R.summarize(v, len(v)) for t, v in by_type.items()},
                                  "latency_ms": {"median": round(statistics.median(lat), 1),
                                                 "p95": round(float(np.percentile(lat, 95)), 1)}}
        print(key, json.dumps(res["conditions"][key]["exact"]), "win", json.dumps(res["conditions"][key]["window"]), flush=True)
res["note_rerank"] = ("hybrid+rerank (no bracket) used the context window as the rerank passage; the next message's "
                      "window contains the target, so it won exact r@1. [msg] / [ctx] fix that.")
json.dump(res, open(R.OUT, "w"), indent=2)
