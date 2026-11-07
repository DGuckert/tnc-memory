"""Conversation entity tracker for reference resolution.

Replaces the gateway's regex shortcuts (find_exact/context/fuzzy_project) with explicit state:

  mentions  every known entity mentioned, with turn index, speaker, and polarity
            (negated mentions like "not Grost" are recorded but excluded as referents)
  intro     order in which entities first appeared  -> "first / original / new / latest / second"
  focus     the entity most recently *talked about* (a resolved query or the last positive
            mention)                                -> "it / that one / that project"
  other     every entity except the focus           -> "the other one" (unique only when
                                                       exactly one non-focus entity exists)

Resolution returns {"entity", "how", "confidence", "candidates"}; entity=None with how="clarify"
means the reference is genuinely ambiguous and the caller should ask, not guess.

Entities are the names the memory store knows about (e.g. projects); unknown capitalised words
(people's names in chatter) are ignored. One- and two-edit typos of known names (OSA distance,
names >= 4 chars) are corrected.
"""
import json, re

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")

NEG = re.compile(r"\b(not|no|isn't|wasn't|except|instead of|rather than|forget|don't mean|dont mean|do not mean|"
                 r"not talking about|other than|besides)\s+(?:(?:for|with|on|about|from)\s+)?(?:the\s+)?(?:project\s+)?$", re.I)
R_OTHER = re.compile(r"\b(the\s+other|other\s+one|other\s+project|the\s+remaining|the\s+rest)\b", re.I)
R_NEW = re.compile(r"\b(new|newest|latest|newer|recent|most\s+recent|last\s+one|just\s+came\s+in|just\s+added|latest\s+one)\b", re.I)
R_FIRST = re.compile(r"\b(first|original|old|older|oldest|earlier|earliest|previous)\b(?!\s+(?:time|week|month))", re.I)
R_SECOND = re.compile(r"\b(second|2nd|middle)\b", re.I)
R_THIRD = re.compile(r"\b(third|3rd)\b", re.I)
R_PRON = re.compile(r"\b(it|its|it's|that\s+one|that\s+project|this\s+one|this\s+project|same\s+one|same\s+project|the\s+project|that\s+job|this\s+job|the\s+job|them)\b", re.I)


def osa(a, b):
    prev2 = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


class EntityTracker:
    def __init__(self, entities, vocab=None, slot_words=None):
        """entities: iterable of canonical names. vocab: set of ordinary lowercase words that
        must never be typo-corrected into an entity. slot_words: attribute names the memory
        store holds (e.g. "access code"); when given, only earlier user turns that ask for one
        of them are treated as memory questions that move the focus - chatter that happens to
        say "it" or "the other" no longer hijacks it."""
        self.canon = {e.lower(): e for e in entities}
        self.vocab = vocab or set()
        self.slot = re.compile(r"\b(" + "|".join(re.escape(s) for s in slot_words) + r")\b", re.I) if slot_words else None
        self._fuzzy_cache = {}

    def _match(self, tok):
        t = tok.lower()
        if t in self.canon:
            return self.canon[t], 0
        if len(t) < 4 or t in self.vocab or not tok[0].isupper() and len(t) < 6:
            return None, None
        if t in self._fuzzy_cache:
            return self._fuzzy_cache[t]
        best = (None, None)
        maxd = 1 if len(t) < 7 else 2
        for c, e in self.canon.items():
            if abs(len(c) - len(t)) > maxd:
                continue
            d = osa(t, c)
            if d <= maxd and (best[1] is None or d < best[1]):
                best = (e, d)
        self._fuzzy_cache[t] = best
        return best

    def mentions_in(self, text):
        out = []
        for m in WORD.finditer(text):
            e, d = self._match(m.group(0))
            if e is None:
                continue
            neg = bool(NEG.search(text[:m.start()][-40:]))
            out.append({"entity": e, "typo": d > 0, "negated": neg, "pos": m.start()})
        return out

    QUESTION = re.compile(r"\?|\b(what|what's|whats|which|pull|get|need|have|gimme|give me|send|look up|find)\b", re.I)

    def track(self, messages):
        """Build state from all turns except the last user turn. Earlier user questions that
        used a reference ("and the other one?") are resolved in sequence, so the focus follows
        the conversation instead of just the last name typed."""
        st = {"intro": [], "mentions": [], "focus": None, "focus_turn": -1}
        for ti, m in enumerate(messages):
            role, text = m.get("role"), m.get("content", "") or ""
            if role == "system":
                continue
            if role == "assistant":
                # a previous resolved tool call sets focus strongly
                try:
                    obj = json.loads(text)
                    p = obj.get("args", {}).get("project")
                    if p and p.lower() in self.canon:
                        e = self.canon[p.lower()]
                        st["focus"], st["focus_turn"] = e, ti
                        if e not in st["intro"]:
                            st["intro"].append(e)
                    continue
                except Exception:
                    pass
            ms = self.mentions_in(text)
            # with known slots, naming one is a request ("same for the other one, the access code");
            # without, fall back to question-shaped turns
            is_q = self.slot.search(text) if self.slot else self.QUESTION.search(text)
            if role == "user" and not any(not x["negated"] for x in ms) and is_q and st["intro"]:
                r = self._resolve(st, text)
                if r["entity"]:
                    st["focus"], st["focus_turn"] = r["entity"], ti
            for mm in ms:
                mm["turn"] = ti
                st["mentions"].append(mm)
                if mm["entity"] not in st["intro"]:
                    st["intro"].append(mm["entity"])
                if not mm["negated"]:
                    st["focus"], st["focus_turn"] = mm["entity"], ti
        return st

    def resolve(self, messages):
        hist, last = messages[:-1], messages[-1].get("content", "") if messages else ""
        return self._resolve(self.track(hist), last)

    def _resolve(self, st, last):
        intro = st["intro"]
        here = self.mentions_in(last)
        pos = [m for m in here if not m["negated"]]
        neg = {m["entity"] for m in here if m["negated"]}
        res = lambda e, how, c, cands=None: {"entity": e, "how": how, "confidence": c,
                                             "candidates": cands or ([e] if e else intro), "state": st}
        # 1. explicit (possibly typo'd) positive mention in the question wins
        if pos:
            e = pos[-1]["entity"]
            return res(e, "typo" if pos[-1]["typo"] else "explicit", 0.9 if pos[-1]["typo"] else 0.99)
        pool = [e for e in intro if e not in neg]
        if neg and len(pool) == 1:
            return res(pool[0], "negation", 0.9)
        if not pool:
            return res(None, "clarify", 0.0, [])
        # 2. ordinal / recency words
        if R_OTHER.search(last):
            focus = st["focus"] if st["focus"] not in neg else None
            cands = [e for e in pool if e != focus]
            if len(cands) == 1:
                return res(cands[0], "other", 0.9)
            if not cands:
                return res(None, "clarify", 0.0, pool)
            # several non-focus entities: if the focus was set by a resolved query, "the other"
            # contrasts with it, but there is still more than one candidate -> ask
            return res(None, "clarify", 0.2, cands)
        if R_SECOND.search(last) and len(pool) >= 2:
            return res(pool[1], "second", 0.85)
        if R_THIRD.search(last) and len(pool) >= 3:
            return res(pool[2], "third", 0.85)
        if R_NEW.search(last):
            return res(pool[-1], "newest", 0.85)
        if R_FIRST.search(last):
            return res(pool[0], "first", 0.85)
        # 3. pronoun / bare reference -> focus
        if R_PRON.search(last) or True:
            f = st["focus"]
            if f and f not in neg:
                n_distinct = len(pool)
                return res(f, "focus", 0.9 if n_distinct == 1 else 0.75)
            if len(pool) == 1:
                return res(pool[0], "only", 0.8)
        return res(None, "clarify", 0.1, pool)
