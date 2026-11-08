"""Measure host prefill / decode speed at each k using llama-server's own timings.

Builds real reader prompts (9.75M store notes for the first 20 benchmark conversations) at
k = 1, 5, 10, 20, 50, plus long synthetic note lists (k = 100, 200) to see where prefill bends.
Prompt cache is disabled per request (cache_prompt=false) so every prompt is a cold prefill.
"""
import json, statistics, sys, urllib.request
sys.path.insert(0, "/home/daniel/tnc")
from memory_api import Memory
from topk_sweep import load_975
from host_reader_sweep import build_msgs, complete

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8131"
TAG = sys.argv[2] if len(sys.argv) > 2 else "gemma3-1b"


def call(msgs):
    _, _, t, _ = complete(URL, msgs, 16, cache_prompt=False)
    return t.get("prompt_n"), t.get("prompt_per_second"), t.get("prompt_ms"), t.get("predicted_per_second")


docs, cases = load_975()
mem = Memory(fact_docs=docs, search_k=200)
cases = cases[:20]
rets = [(c, mem.retrieve([m for m in c[1] if m["role"] != "system"], k=200, sources=("fact",))) for c in cases]
out = {"host": TAG, "rows": []}
KS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 5, 10, 20, 50, 100]
for k in KS:
    pn, pps, pms, dps = [], [], [], []
    for (kind, conv, gold), r in rets:
        a, b, c, d = call(build_msgs(conv, r, r["notes"][:k]))
        pn.append(a); pps.append(b); pms.append(c); dps.append(d)
    row = {"k": k, "prompt_tokens": statistics.median(pn), "prefill_tok_s": round(statistics.median(pps)),
           "prefill_ms": round(statistics.median(pms), 1), "decode_tok_s": round(statistics.median(dps))}
    out["rows"].append(row)
    print(json.dumps(row), flush=True)
json.dump(out, open(f"/home/daniel/tnc/work/prefill_{TAG}.json", "w"), indent=2)
