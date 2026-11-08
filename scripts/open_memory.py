"""Open-ended memory: free-form facts about anyone or anything, with the same write-time properties as the
typed project store (dated history, newer state replaces older, grounded in the message, cards).

A fact is (subject, relation, value, kind, date, src):
  kind 'state'  has one current value that can change: where someone lives, an editor, a deadline
  kind 'note'   accumulates: reasons, incidents, one-off events

Consolidation (consolidate_open):
  - subject resolution: the user is 'me' (also 'I', 'daniel'); when another speaker says 'I' the extractor
    names them; a subject must be the speaker, 'me' from the user, or named in the message or the
    same author's adjacent messages
  - grounding: a value must share a content word (or number) with the message window
  - relation normalisation: per subject, a state relation joins an existing one when their bge embeddings
    are close ('based in' ~ 'lives in' ~ 'moved to'), so a change supersedes instead of piling up
  - duplicates (same subject, relation, value, message) collapse
Retrieval: subject cards for subjects the question names, plus hybrid search over the fact sentences.
"""
import re
from collections import defaultdict

STOP = set("the a an and or of to in on for is it that this was with at as by from my your their our his her its "
           "just now then so be been are were has have had i me you he she they we".split())
ME = {"me", "i", "myself", "daniel", "user", "the user"}


def _content(s):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9+#.\-']*", s.lower()) if len(w) >= 3 and w not in STOP}


class OpenStore:
    def __init__(self, embed=None, rel_threshold=0.80):
        self.facts = []              # dicts: subj, rel, relkey, val, kind, date, src
        self.rels = defaultdict(dict)  # subj -> relkey -> embedding
        self.embed = embed
        self.th = rel_threshold

    def _relkey(self, subj, rel, kind):
        rel = re.sub(r"\s+", " ", rel.lower().strip())
        if kind != "state":
            return rel
        known = self.rels[subj]
        if rel in known:
            return rel
        if self.embed is not None and known:
            import numpy as np
            v = self.embed(rel)
            best, sim = None, -1
            for k, kv in known.items():
                s = float(np.dot(v, kv))
                if s > sim:
                    best, sim = k, s
            if sim >= self.th:
                return best
            known[rel] = v
            return rel
        known[rel] = self.embed(rel) if self.embed else None
        return rel

    def add(self, subj, rel, val, kind, date, src):
        key = self._relkey(subj, rel, kind)
        if kind == "state":
            h = self.history(subj, key)
            if h and h[-1]["val"].lower() == val.lower():
                return False  # restates the current value; returning to an older value is a real change
        elif any(f["subj"] == subj and f["relkey"] == key and f["val"].lower() == val.lower() and f["src"] == src
                 for f in self.facts):
            return False
        self.facts.append({"subj": subj, "rel": rel, "relkey": key, "val": val, "kind": kind, "date": date, "src": src})
        return True

    def subjects(self):
        return sorted({f["subj"] for f in self.facts})

    def history(self, subj, relkey):
        return sorted([f for f in self.facts if f["subj"] == subj and f["relkey"] == relkey and f["kind"] == "state"],
                      key=lambda f: (f["date"], f["src"]))

    def card(self, subj):
        fs = [f for f in self.facts if f["subj"] == subj]
        parts = []
        for key in sorted({f["relkey"] for f in fs if f["kind"] == "state"}):
            h = self.history(subj, key)
            cur = f"{key}: {h[-1]['val']} (since {h[-1]['date']})"
            if len(h) > 1:
                cur += "; before that " + ", ".join(f"{x['val']} (from {x['date']})" for x in reversed(h[:-1]))
            parts.append(cur)
        notes = sorted([f for f in fs if f["kind"] == "note"], key=lambda f: f["date"])
        if notes:
            parts.append("notes: " + "; ".join(f"[{f['date']}] {f['rel']}: {f['val']}" for f in notes))
        name = "the user (me)" if subj == "me" else subj
        return f"{name} | " + " | ".join(parts)

    def sentences(self):
        """One searchable sentence per fact; superseded states are marked so search can't serve them as current."""
        out = []
        latest = {}
        for f in self.facts:
            if f["kind"] == "state":
                k = (f["subj"], f["relkey"])
                if k not in latest or (f["date"], f["src"]) >= (latest[k]["date"], latest[k]["src"]):
                    latest[k] = f
        for i, f in enumerate(self.facts):
            old = f["kind"] == "state" and latest[(f["subj"], f["relkey"])] is not f
            who = "the user" if f["subj"] == "me" else f["subj"]
            out.append((i, f"{who} {f['rel']}: {f['val']} ({f['date']}{', no longer current' if old else ''})"))
        return out


TYPED_REL = re.compile(r"\b(port|language|written in|host|hosted|runs on|running on|deployed|database|db|talks to|"
                       r"storage|status|on hold|paused|shipped|abandoned|resum|collaborat|helping|joined|project name|"
                       r"started|moved from|moved to|moved off|lives on)\b")


def consolidate_open(raw, msgs, embed=None, self_names=("daniel",), hygiene=True, stats=None, typed_entities=()):
    """raw entries [j, subject, 'open', kind, 'relation|value'] -> OpenStore.
    typed_entities: projects held by the typed store; open facts restating their typed settings are dropped
    (the open pass re-records 'Tundrel port 4687' despite being told not to, and those copies go stale)."""
    st = stats if stats is not None else {}
    store = OpenStore(embed=embed)
    typed = {e.lower() for e in typed_entities}
    for j, subj, attr, kind, rv in sorted([r for r in raw if r[2] == "open"], key=lambda r: r[0]):
        if "|" not in rv:
            continue
        rel, val = [x.strip() for x in rv.split("|", 1)]
        subj = subj.strip()
        author = msgs[j]["a"]
        if subj.lower() in ME:
            subj = "me" if author.lower() in self_names else author
        if hygiene:
            window = " ".join(msgs[i]["t"] for i in range(max(0, j - 2), min(len(msgs), j + 2))
                              if msgs[i]["a"] == author or i == j)
            low = window.lower()
            ok_subj = (subj == "me" and author.lower() in self_names) or subj.lower() == author.lower() or \
                subj.lower() in low
            if not ok_subj:
                st["open_unseen_subject"] = st.get("open_unseen_subject", 0) + 1
                continue
            # at least half the value's content words must be in the message window: sharing one word
            # ('kestrel') let 'kestrel was too full' stand in for 'kestrel kept overheating'
            vc = _content(val)
            if vc and len(vc & _content(window)) * 2 < len(vc):
                st["open_ungrounded"] = st.get("open_ungrounded", 0) + 1
                continue
            if subj.lower() in typed and TYPED_REL.search(rel.lower()) and "reason" not in rel.lower():
                st["open_typed_dupe"] = st.get("open_typed_dupe", 0) + 1
                continue
        if store.add(subj, rel, val, kind, msgs[j]["date"], j):
            st["open_kept"] = st.get("open_kept", 0) + 1
        else:
            st["open_dupe"] = st.get("open_dupe", 0) + 1
    return store


def oracle_open(open_events, msgs, embed=None):
    """Perfect extraction of the open events (relations named the way the generator meant them)."""
    store = OpenStore(embed=embed)
    src = {}
    for j, m in enumerate(msgs):
        if m.get("oev") is not None:
            src.setdefault(m["oev"], j)
    for i, e in enumerate(open_events):
        j = src[i]
        d = msgs[j]["date"]
        t = e["type"]
        if t == "person":
            store.add(e["subject"], e["rel"], e["value"], "note" if e["rel"] == "pet" else "state", d, j)
        elif t == "pref":
            store.add("me", e["rel"], e["value"], "state", d, j)
        elif t == "reason":
            a, v, o = e["about"]
            what = {"status": f"reason for {v}"}.get(a, f"reason for changing {a}" + (f" from {o}" if o else ""))
            store.add(e["subject"], what, e["text"], "note", d, j)
        elif t == "plan":
            store.add(e["subject"], "plan", e["text"], "note", d, j)
        else:
            store.add(e["subject"], "incident", e["text"], "note", d, j)
    return store


def mentioned_subjects(question, store, extra_names=()):
    q = question.lower()
    subs = [s for s in store.subjects() if s != "me" and re.search(r"(?<![\w])" + re.escape(s.lower()) + r"(?![\w])", q)]
    if re.search(r"\b(i|my|me|mine|i'm)\b", q) and "me" in store.subjects():
        subs.append("me")
    return subs
