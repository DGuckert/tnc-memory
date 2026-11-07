"""Re-score the tracker after the post-hoc negation fix ("I don't mean X", "other than", "besides").
The held-out set is no longer blind for this number; the frozen result stays in resolve_metrics.json."""
import json, sys
sys.path.insert(0, "/home/daniel/tnc")
sys.path.insert(0, "/home/daniel/tnc-repo/scripts")
import resolve_eval as R
from entity_tracker import EntityTracker

v = R.word_vocab()
out = {}
for ds in ("hard", "reference", "heldout"):
    names, docs, rows = R.load(ds)
    tr = EntityTracker(names, vocab=v, slot_words=R.SLOTS)
    out[ds] = round(sum(tr.resolve(r["messages"])["entity"] == r["gold"] for r in rows) / len(rows), 3)
print(out)
json.dump({"tracker_posthoc_negation_fix": out,
           "note": "after adding don't mean / other than / besides to NEG; held-out is not blind for this number"},
          open("/home/daniel/tnc/work/reference_eval/resolve_posthoc.json", "w"), indent=1)
