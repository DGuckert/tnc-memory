"""Evaluate message retrieval on the hand-written real-message eval (205 questions, 603K messages).

Hit definitions
  exact    the target message itself
  window   target, or a message within +-2 positions in the same channel (same exchange)

Conditions: bm25 / dense_msg / dense_ctx / hybrid (RRF) / hybrid+rerank, each global and with the
scope the question implies (author for type A, channel for type C; V/E questions are unscoped,
so the 'scoped' column only differs for A/C rows). Also a 'channel-known' oracle where every
query is restricted to its target channel, to show what channel scoping alone buys.
"""
import json, os, statistics, sys, time
import numpy as np
sys.path.insert(0, "/home/daniel/tnc")
from message_index import MessageIndex

EVAL = "/home/daniel/tnc/work/real_eval/real_eval.jsonl"
OUT = "/home/daniel/tnc/work/real_eval/real_eval_metrics.json"


ORD = None  # ordinal of each message within its channel


def hits(ret, tgt, msgs):
    ex = win = None
    for r, i in enumerate(ret):
        if ex is None and i == tgt:
            ex = r + 1
        if win is None and msgs[i]["ch"] == msgs[tgt]["ch"] and abs(ORD[i] - ORD[tgt]) <= 2:
            win = r + 1
    return ex, win


def summarize(ranks, n):
    out = {}
    for k in (1, 5, 10, 50):
        out[f"r@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 3)
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
    return out


def main():
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    t0 = time.time()
    idx = MessageIndex()
    print("index ready", round(time.time() - t0, 1), "s", flush=True)
    msgs = idx.msgs
    global ORD
    cnt = {}
    ORD = []
    for m in msgs:
        ORD.append(cnt.get(m["ch"], 0)); cnt[m["ch"]] = ORD[-1] + 1
    conds = [("bm25", "bm25", False), ("dense_msg", "dense_msg", False), ("dense_ctx", "dense_ctx", False),
             ("hybrid", "hybrid", False), ("hybrid+rerank", "hybrid", True)]
    res = {"n": len(rows), "n_messages": len(msgs), "conditions": {}}
    per_q = []
    for name, mode, rr in conds:
        for scope in ("global", "scoped", "channel_oracle"):
            ex, win, lat = [], [], []
            by_type = {}
            for r in rows:
                kw = {}
                if scope == "scoped":
                    if r["type"] == "A":
                        kw["author"] = r["author"]
                    if r["type"] == "C":
                        kw["channel"] = r["channel"]
                if scope == "channel_oracle":
                    kw["channel"] = r["channel"]
                t = time.perf_counter()
                ret = [i for i, _ in idx.search(r["question"], k=50, mode=mode, rerank=rr, **kw)]
                lat.append((time.perf_counter() - t) * 1000)
                e, w = hits(ret, r["target_idx"], msgs)
                ex.append(e); win.append(w)
                by_type.setdefault(r["type"], []).append(e)
                if name == "hybrid+rerank" and scope == "scoped":
                    per_q.append({"id": r["id"], "type": r["type"], "exact_rank": e, "window_rank": w,
                                  "top1": ret[0] if ret else None})
            n = len(rows)
            key = f"{name}|{scope}"
            res["conditions"][key] = {
                "exact": summarize(ex, n), "window": summarize(win, n),
                "exact_by_type": {t: summarize(v, len(v)) for t, v in by_type.items()},
                "latency_ms": {"median": round(statistics.median(lat), 1),
                               "p95": round(float(np.percentile(lat, 95)), 1)},
            }
            print(key, json.dumps(res["conditions"][key]["exact"]), "win", json.dumps(res["conditions"][key]["window"]),
                  res["conditions"][key]["latency_ms"], flush=True)
    res["per_question"] = per_q
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
