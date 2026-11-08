"""Where does extraction still go wrong after hygiene? Compare the consolidated store with the oracle,
and show the messages behind each wrong attribute."""
import json, sys
from lifelog_memory import consolidate, oracle_store

D = sys.argv[1] if len(sys.argv) > 1 else "/home/daniel/tnc/work/lifelog/small"
ev = [tuple(e) for e in json.load(open(f"{D}/events.json"))]
msgs = [json.loads(l) for l in open(f"{D}/log.jsonl")]
o = oracle_store(ev)
raw = json.load(open(f"{D}/extracted_raw.json"))["raw"]
fs = consolidate(raw, msgs)
for e in o.entities():
    for a in ("started", "language", "database", "host", "port", "status"):
        gh = [str(v) for _, v in o.history(e, a)]
        hh = [str(v) for _, v in fs.history(e, a)]
        if [x.lower() for x in gh] != [x.lower() for x in hh]:
            print(f"{e:9s} {a:9s} gold={gh}  got={hh}")
            # the event messages for this entity/attribute
            for i, (d, p, aa, op, v, old, _) in enumerate(ev):
                if p == e and aa == a:
                    for j, m in enumerate(msgs):
                        if m["ev"] == i:
                            got = [r for r in raw if r[0] == j]
                            print(f"     msg {j} [{m['date']}] {m['t']!r} -> raw {got}")
    gc, hc = sorted(o.collaborators(e)), sorted(fs.collaborators(e))
    if [x.lower() for x in gc] != [x.lower() for x in hc]:
        print(f"{e:9s} collab    gold={gc}  got={hc}")
