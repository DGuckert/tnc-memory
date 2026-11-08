"""Store accuracy for small and big logs with the current consolidation code (no LLM needed)."""
import json
from lifelog_memory import consolidate, oracle_store, store_accuracy

for name in ("small", "big"):
    D = f"/home/daniel/tnc/work/lifelog/{name}"
    msgs = [json.loads(l) for l in open(f"{D}/log.jsonl")]
    raw = json.load(open(f"{D}/extracted_raw.json"))["raw"]
    ev = [tuple(e) for e in json.load(open(f"{D}/events.json"))]
    st = {}
    fs = consolidate(raw, msgs, stats=st)
    st.pop("reattach_log", None)
    acc = store_accuracy(fs, oracle_store(ev))
    print(name, acc["current_value_acc"], acc["history_exact"], "spurious:", acc["spurious_entities"],
          "| missing:", [e for e in oracle_store(ev).entities() if e not in fs.entities()])
    print("   ", st)
