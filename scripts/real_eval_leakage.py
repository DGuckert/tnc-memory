"""How much of the real-eval score is word overlap between the hand-written question and the target?

Buckets questions by the number of shared content words (lowercased, stopwords removed, len>=3) and
reports exact r@1 / r@10 per bucket for bm25, dense_msg and hybrid+rerank[msg].
"""
import json, re, sys
sys.path.insert(0, "/home/daniel/tnc")
import real_eval_run as R
from message_index import MessageIndex

STOP = set("the a an and or of to in on for is it that this what was who how did does do with about someone say said "
           "saying be are i you they he she we my your their at as by from but not so if just like any there which when "
           "where why can could would should me its than then them too very get got use using".split())
cw = lambda s: {w for w in re.findall(r"[a-z0-9][a-z0-9.\-]*", s.lower()) if len(w) >= 3 and w not in STOP}
rows = [json.loads(l) for l in open(R.EVAL, encoding="utf-8")]
idx = MessageIndex()
out = {}
for r in rows:
    r["overlap"] = len(cw(r["question"]) & cw(idx.msgs[r["target_idx"]]["t"]))
buckets = {"0-1": lambda o: o <= 1, "2-3": lambda o: 2 <= o <= 3, "4+": lambda o: o >= 4}
for name, kw in (("bm25", dict(mode="bm25", rerank=False)), ("dense_msg", dict(mode="dense_msg", rerank=False)),
                 ("hybrid+rerank[msg]", dict(mode="hybrid", rerank=True, rerank_passage="msg"))):
    ranks = []
    for r in rows:
        ret = [i for i, _ in idx.search(r["question"], k=10, **kw)]
        ranks.append(ret.index(r["target_idx"]) + 1 if r["target_idx"] in ret else None)
    out[name] = {}
    for b, f in buckets.items():
        sel = [x for x, r in zip(ranks, rows) if f(r["overlap"])]
        out[name][b] = {"n": len(sel), "r@1": round(sum(1 for x in sel if x == 1) / max(1, len(sel)), 3),
                        "r@10": round(sum(1 for x in sel if x) / max(1, len(sel)), 3)}
    print(name, out[name], flush=True)
res = json.load(open(R.OUT))
res["overlap_buckets"] = out
json.dump(res, open(R.OUT, "w"), indent=2)
