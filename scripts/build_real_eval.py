"""Join hand-written questions (q1-3.txt: cand_no|type|question) with candidates -> real_eval.jsonl.

type: V vague topic recall, E names an entity (model/tool/hardware), A author-scoped
(the question names the author; the eval also passes author= as a scope filter),
C channel-scoped (passes channel=).
"""
import json, re, sys

cands = [json.loads(l) for l in open("/home/daniel/tnc/work/real_eval/candidates.jsonl", encoding="utf-8")]
out = []
for fn in sys.argv[1:]:
    for line in open(fn, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        n, typ, q = line.split("|", 2)
        c = cands[int(n)]
        row = {"id": f"real-{int(n):03d}", "type": typ, "question": q, "target_idx": c["idx"],
               "channel": c["ch"], "author": c["a"]}
        out.append(row)
with open("/home/daniel/tnc/work/real_eval/real_eval.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
from collections import Counter
print(len(out), Counter(o["type"] for o in out))
