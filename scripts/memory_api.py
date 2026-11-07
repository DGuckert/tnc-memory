"""TNC memory middleman, v2 interface: return notes, not answers.

    mem = Memory(fact_docs=docs, message_index=MessageIndex(), code_index=CodeIndex(...))
    out = mem.retrieve(messages, k=5)

out = {
  "entity":      resolved project / entity name or None,
  "resolution":  {"how": "explicit|typo|focus|other|newest|first|...|clarify", "confidence", "candidates"},
  "needs_clarification": bool,          # True -> host should ask, not answer
  "clarify":     a ready-made question when needs_clarification,
  "structured":  {"slot", "value"} when a typed fact matches exactly (no model needed),
  "notes":       [{"text", "source": "fact|message|code", "score", "ref"}],
  "context":     the notes formatted as a compact block for the host model's prompt,
  "timings_ms":  per-stage latency,
}

The host model (any size) does the reading. The middleman's job is: resolve the reference,
scope the search, rank, and keep the context small. The old 360M reader is no longer in the path.
"""
import re, time
from entity_tracker import EntityTracker

SLOT_RE = re.compile(r"\b(access code|invoice number|serial number|passphrase|license key|tracking id)\b", re.I)


VOCAB_PATH = "/home/daniel/tnc/work/vocab_min20.txt"


def general_vocab(path=VOCAB_PATH, corpus="/home/daniel/tnc/corpus.jsonl", min_count=20):
    """Words seen >= min_count times in the chat corpus. A small fact store's own vocabulary is
    not enough: with 80 notes, "This" was one edit from the project "Thias" and got corrected."""
    import json, os, collections
    if os.path.exists(path):
        return set(open(path).read().split())
    if not os.path.exists(corpus):
        return set()
    c = collections.Counter()
    for l in open(corpus, encoding="utf-8"):
        c.update(re.findall(r"[a-z]+", json.loads(l)["t"].lower()))
    words = {w for w, n in c.items() if n >= min_count}
    open(path, "w").write("\n".join(sorted(words)))
    return words


def extract_note(text):
    m = re.findall(r"Note to self:.*", text)
    return m[-1].strip() if m else text.strip()


class Memory:
    def __init__(self, fact_docs=None, message_index=None, code_index=None, vocab=None,
                 slot_words=None, reranker=True, search_k=50):
        self.fact_gw = None
        entities = []
        if fact_docs is not None:
            from gateway import MemoryGateway
            self.fact_gw = MemoryGateway(fact_docs, llms=False, structured=True, search_k=search_k)
            entities = list(self.fact_gw.entity_map.values())
            if vocab is None:  # ordinary words are never typo-corrected into names
                vocab = set(self.fact_gw.postings) | general_vocab()
                vocab -= set(self.fact_gw.entity_map)
        self.msg = message_index
        self.code = code_index
        if code_index is not None:
            entities += code_index.entity_names()
        self.tracker = EntityTracker(entities, vocab=vocab,
                                     slot_words=slot_words or ["access code", "invoice number", "serial number",
                                                               "passphrase", "license key", "tracking id"])

    # ---------- helpers ----------
    @staticmethod
    def format_context(notes):
        lines = []
        for i, n in enumerate(notes, 1):
            tag = n["source"]
            if tag == "message":
                lines.append(f"[{i}] (#{n['ref']['channel']} {n['ref']['author']}) {n['text']}")
            elif tag == "code":
                lines.append(f"[{i}] ({n['ref']['repo']}:{n['ref']['path']}) {n['text']}")
            else:
                lines.append(f"[{i}] {n['text']}")
        return "\n".join(lines)

    def _fact_notes(self, query, entity, k):
        gw = self.fact_gw
        q = query if not entity or entity.lower() in query.lower() else f"{query} project {entity}"
        ids, mode = gw.search(q, entity, k=max(gw.search_k, k))
        canon0 = gw.docs[int(ids[0])]
        from gateway import canonical_from_note
        rq = canonical_from_note(extract_note(canon0)) or q
        ids = gw.rerank(rq, ids, k)
        return [{"text": extract_note(gw.docs[int(i)]), "source": "fact", "score": None, "ref": {"doc": int(i)}}
                for i in ids]

    # ---------- main entry ----------
    def retrieve(self, messages, k=5, sources=("fact", "message", "code"), scope=None):
        t0 = time.perf_counter()
        scope = scope or {}
        last = messages[-1].get("content", "") if messages else ""
        r = self.tracker.resolve(messages)
        t1 = time.perf_counter()
        out = {"entity": r["entity"], "resolution": {k_: r[k_] for k_ in ("how", "confidence", "candidates")},
               "needs_clarification": False, "clarify": None, "structured": None, "notes": [], "context": ""}
        slot = SLOT_RE.search(last)
        wants_fact = bool(slot) and self.fact_gw is not None and "fact" in sources
        if wants_fact and r["entity"] is None:
            out["needs_clarification"] = True
            c = r["candidates"]
            out["clarify"] = (f"Which project do you mean: {', '.join(c)}?" if c else "Which project do you mean?")
            out["timings_ms"] = {"resolve": (t1 - t0) * 1e3, "total": (time.perf_counter() - t0) * 1e3}
            return out
        notes = []
        if wants_fact:
            key = (r["entity"].lower(), slot.group(1).lower())
            v = self.fact_gw.fact_map.get(key)
            if v:
                out["structured"] = {"entity": r["entity"], "slot": slot.group(1), "value": v}
            q = f"what is the {slot.group(1)} for project {r['entity']}?"
            notes += self._fact_notes(q, r["entity"], k)
        t2 = time.perf_counter()
        if self.msg is not None and "message" in sources and not wants_fact:
            for i, s in self.msg.search(last, k=k, channel=scope.get("channel"), author=scope.get("author"),
                                        time=scope.get("time")):
                n = self.msg.note(i)
                notes.append({"text": n["text"], "source": "message", "score": s,
                              "ref": {"idx": i, "channel": n["channel"], "author": n["author"]}})
        if self.code is not None and "code" in sources and not wants_fact:
            repo = scope.get("repo") or (r["entity"] if r["entity"] in self.code.repos else None)
            for hit in self.code.search(last, k=k, repo=repo):
                notes.append({"text": hit["summary"], "source": "code", "score": hit["score"], "ref": hit["ref"]})
        t3 = time.perf_counter()
        out["notes"] = notes[: k if len(sources) == 1 else 2 * k]
        out["context"] = self.format_context(out["notes"])
        out["timings_ms"] = {"resolve": (t1 - t0) * 1e3, "facts": (t2 - t1) * 1e3, "open": (t3 - t2) * 1e3,
                             "total": (t3 - t0) * 1e3}
        return out
