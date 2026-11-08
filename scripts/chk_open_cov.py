"""Which open events did extraction capture? Per type: covered = the event message produced an open fact."""
import json, sys
from collections import Counter
D = sys.argv[1] if len(sys.argv) > 1 else "/home/daniel/tnc/work/lifelog/small2"
msgs = [json.loads(l) for l in open(f"{D}/log.jsonl")]
oev = json.load(open(f"{D}/open_events.json"))
d = json.load(open(f"{D}/extracted_raw.json"))
raw = d["raw"] + d.get("recall", {}).get("raw", [])
by_msg = {}
for r in raw:
    by_msg.setdefault(r[0], []).append(r)
tot, cov = Counter(), Counter()
for j, m in enumerate(msgs):
    if m.get("oev") is None:
        continue
    t = oev[m["oev"]]["type"]
    tot[t] += 1
    got = by_msg.get(j, [])
    if any(r[2] == "open" for r in got):
        cov[t] += 1
    elif len([x for x in tot.elements() if x == t]) <= 400:
        print(f"MISS {t:8s} [{m['a']}] {m['t'][:70]!r} -> {got}")
print({t: f"{cov[t]}/{tot[t]}" for t in tot})
