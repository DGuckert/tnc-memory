"""Write-time consolidation for the life log: LLM fact extraction -> fact store with history -> entity cards.

The point: a top-k retriever cannot answer "which projects use Postgres now?" or "what port was it on
before?", because the answer is spread over many messages and later messages supersede earlier ones.
So instead of only indexing messages, the memory layer READS every message once as it arrives,
extracts typed facts, and keeps a small state that is always current and keeps its history:

  FactStore   (entity, attribute) -> [(date, value, source msg)], collaborators as add/remove
  card()      one compact paragraph per project: current values + what they were before + dates
  query()     aggregate lookups: which entities have attr == value (now, or as of a date)

Extraction uses the host LLM (llama-server, raw ChatML, thinking off) over batches of messages, with
the preceding messages as context so "moved it to basalt" resolves. The attribute schema is fixed
(project memory); values are free text. An oracle store built from the gold events gives the upper
bound, so extraction errors and reasoning errors can be told apart.
"""
import json, re, sys, time
sys.path.insert(0, "/home/daniel/tnc")

ATTRS = ("started", "language", "database", "host", "port", "status", "collaborator")
STATUS = ("active", "paused", "shipped", "abandoned")


class FactStore:
    def __init__(self):
        self.h = {}          # (entity, attr) -> list of (date, value, src)
        self.collab = {}     # entity -> list of (date, op, person, src)
        self.names = {}      # lower -> canonical entity name

    def canon(self, e):
        e = e.strip().strip("\"'")
        k = e.lower()
        if k not in self.names:
            self.names[k] = e
        return self.names[k]

    def apply(self, entity, attr, op, value, date, src=None):
        if attr not in ATTRS or not entity or value in (None, ""):
            return
        v = str(value).strip()
        if v.lower() in ("unknown", "none", "n/a", "na", "null", "tbd", "?", "-"):
            return
        e = self.canon(entity)
        if attr == "status":
            v = v.lower()
            if v not in STATUS:
                return
        if attr == "collaborator":
            self.collab.setdefault(e, []).append((date, "remove" if op == "remove" else "add", v, src))
            return
        h = self.h.setdefault((e, attr), [])
        if not h or h[-1][1].lower() != v.lower():
            h.append((date, v, src))
        if attr != "status" and (e, "status") not in self.h:
            self.h[(e, "status")] = [(date, "active", src)]

    def entities(self, include_provisional=False):
        """Confirmed entities only: introduced (has 'started') or backed by at least 2 non-status facts.
        A name that one stray chatter message attached a fact to stays provisional and out of cards,
        views, counts and queries (real chatter produced 'Atlas' and 'host' this way)."""
        all_ = sorted({e for e, _ in self.h} | set(self.collab))
        if include_provisional:
            return all_
        if any(a == "started" for _, a in self.h):
            # once introductions are being recorded, a project is something that was introduced; on the
            # 600K-token log, chatter attached 2+ facts to Qwen, vllm, OpenRouter and 27 other names
            return [e for e in all_ if (e, "started") in self.h]
        out = []
        for e in all_:
            n = sum(len(h) for (ee, a), h in self.h.items() if ee == e and a != "status") + len(self.collab.get(e, []))
            if n >= 2:
                out.append(e)
        return out

    def value(self, e, attr, as_of=None):
        h = [x for x in self.h.get((e, attr), []) if as_of is None or x[0] <= as_of]
        return h[-1][1] if h else None

    def collaborators(self, e, as_of=None):
        s = []
        for d, op, p, _ in sorted(self.collab.get(e, []), key=lambda x: x[0]):
            if as_of and d > as_of:
                break
            if op == "add" and p not in s:
                s.append(p)
            elif op == "remove" and p in s:
                s.remove(p)
        return s

    def history(self, e, attr):
        if attr == "collaborator":
            return [(d, f"{op} {p}") for d, op, p, _ in self.collab.get(e, [])]
        return [(d, v) for d, v, _ in self.h.get((e, attr), [])]

    def query(self, attr, value, as_of=None):
        out = []
        for e in self.entities():
            if attr == "collaborator":
                ok = any(value.lower() == p.lower() for p in self.collaborators(e, as_of))
            else:
                v = self.value(e, attr, as_of)
                ok = v is not None and v.lower() == str(value).lower()
            if ok:
                out.append(e)
        return out

    def card(self, e):
        parts = []
        st = self.value(e, "started")
        parts.append(f"{e} (started {st})" if st else e)
        for a in ("language", "database", "host", "port", "status"):
            h = self.history(e, a)
            if not h:
                continue
            cur = f"{a}: {h[-1][1]}" + (f" since {h[-1][0]}" if len(h) > 1 or a == "status" else "")
            if len(h) > 1:
                cur += "; before that " + ", ".join(f"{v} (from {d})" for d, v in reversed(h[:-1]))
            parts.append(cur)
        c = self.collaborators(e)
        hist = self.collab.get(e, [])
        if hist:
            parts.append("collaborators now: " + (", ".join(c) if c else "none") + " [" +
                         "; ".join(f"{p} {'joined' if op == 'add' else 'left'} {d}" for d, op, p, _ in hist) + "]")
        return " | ".join(parts)

    def cards(self, ents=None):
        return "\n".join(self.card(e) for e in (ents or self.entities()))

    def views(self):
        """Write-time aggregate views over the current state, so 'which / how many / which first'
        questions become lookups instead of the model scanning every card in its head."""
        ents = self.entities()
        out = []
        for a in ("status", "database", "language", "host"):
            groups = {}
            for e in ents:
                v = self.value(e, a)
                if v is not None:
                    groups.setdefault(v, []).append(e)
            out.append(f"Projects by current {a}: " + "; ".join(
                f"{v} ({len(es)}): {', '.join(es)}" for v, es in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))))
        ports = [(self.value(e, "port"), e) for e in ents if self.value(e, "port")]
        out.append("Current port -> project: " + "; ".join(f"{p}: {e}" for p, e in sorted(ports)))
        people = {}
        for e in ents:
            for p in self.collaborators(e):
                people.setdefault(p, []).append(e)
        if people:
            out.append("Collaborators now -> projects: " + "; ".join(f"{p}: {', '.join(es)}" for p, es in sorted(people.items())))
        starts = sorted((self.value(e, "started") or "9999", e) for e in ents)
        out.append("Projects in the order they were started: " + ", ".join(f"{e} ({d})" for d, e in starts))
        out.append(f"Total projects: {len(ents)}")
        return "\n".join(out)


def oracle_store(events):
    fs = FactStore()
    for d, p, a, op, v, o, _ in events:
        fs.apply(p, "collaborator" if a == "collab" else a, op, v, d)
    return fs


# ---------------- raw llama-server call ----------------
def llm(url, msgs, n_predict, grammar=None, prefix=""):
    """ChatML prompt with an empty closed think block, sent to /completion.
    Qwen3 Thinking checkpoints still reason out loud in the answer, so callers either constrain the
    output with a GBNF grammar or ask for a final 'ANSWER:' line. Returns (text, prompt_tokens, ms)."""
    import urllib.request
    from host_reader_sweep import chatml
    body = {"prompt": chatml(msgs) + prefix, "n_predict": n_predict, "temperature": 0, "cache_prompt": True,
            "stop": ["<|im_end|>"]}
    if grammar:
        body["grammar"] = grammar
    t = time.perf_counter()
    o = json.loads(urllib.request.urlopen(urllib.request.Request(
        url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
    return o["content"], o.get("tokens_evaluated"), (time.perf_counter() - t) * 1e3


def fact_grammar(n):
    """One line per message, in order: 'k: -' or 'k: Project attr=value; Project attr=value'.
    A fixed line count makes repetition loops impossible and keeps decode short."""
    lines = " ".join(f'"{k}: " body "\\n"' for k in range(n))
    return (f"root ::= {lines}\n"
            'body ::= "-" | fact ("; " fact){0,3}\n'
            'fact ::= name " " attr op val\n'
            "name ::= [A-Za-z0-9_]{1,30}\n"
            "val  ::= [A-Za-z0-9_+.:#/-]{1,30}\n"
            'attr ::= "started" | "language" | "database" | "host" | "port" | "status" | "collaborator"\n'
            'op   ::= "=" | "+=" | "-="\n')


# ---------------- LLM extraction ----------------
EXTRACT_SYS = (
    "You maintain a structured memory of the user's software projects. For each NEW message, write one line "
    "'<number>: ' followed by the project facts that message states, or '-' if it states none. A fact is "
    "'<Project> <attr>=<value>'; several facts are separated by '; '. Attributes: started (value = the message "
    "date; only when a project is first introduced), language, database, host (machine name), port (number only), "
    "status (active; paused = stopped for now and expected to resume; shipped = released; abandoned = given up for "
    "good), collaborator (write 'collaborator+=<person>' when someone "
    "joins, 'collaborator-=<person>' when they leave). 'it' means the project most recently discussed in the "
    "context or earlier new messages. Resuming or picking a project back up means status=active. Only facts the "
    "message explicitly states: never guess, never write unknown values. Chatter unrelated to these projects gets '-'.")

GATE = re.compile(r"\b(project|repo|port|:\d{4}\b|postgres|sqlite|redis|mongo|mariadb|rust|golang|go\b|python|"
                  r"typescript|kotlin|c\+\+|moved|migrat|deploy|host|box|server|paus|resum|ship|releas|abandon|"
                  r"dead end|on ice|on hold|helping|joined|left|dropped|rewrote|kicked off|spun up|side project|"
                  r"new thing|it's on|listens)", re.I)


def gate(msg, known):
    t = msg["t"]
    if any(k in t.lower() for k in known):
        return True
    return bool(GATE.search(t))


FACT_RE = re.compile(r"([A-Za-z0-9_]+) (started|language|database|host|port|status|collaborator)(\+=|-=|=)([^;]+)")


def parse_facts(text, batch):
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+):\s*(.*)", line)
        if not m or not (0 <= int(m.group(1)) < len(batch)):
            continue
        for f in FACT_RE.finditer(m.group(2)):
            op = {"=": "set", "+=": "add", "-=": "remove"}[f.group(3)]
            out.append((batch[int(m.group(1))], {"project": f.group(1), "attr": f.group(2), "op": op,
                                                  "value": f.group(4).strip()}))
    return out


def extract(msgs, url, batch_size=10, ctx_n=4, use_gate=False, log=None):
    """msgs: list of dicts with ch/a/t/date. Returns FactStore plus stats."""
    fs = FactStore()
    known = set()
    idx = list(range(len(msgs)))
    if use_gate:
        keep = set()
        for i in idx:
            if gate(msgs[i], known) or (i > 0 and (i - 1) in keep and msgs[i]["a"] == msgs[i - 1]["a"]):
                keep.add(i)
            # learn names introduced in gated messages cheaply: capitalised words after "called"/"thing:"
            for mm in re.finditer(r"(?:called|thing:|repo for|kicked off)\s+([A-Z][a-z]+)", msgs[i]["t"]):
                known.add(mm.group(1).lower())
        idx = sorted(keep | {j + 1 for j in keep if j + 1 < len(msgs) and re.search(r"\bit\b", msgs[j + 1]["t"])})
    stats = {"messages": len(msgs), "sent_to_extractor": len(idx), "calls": 0, "facts": 0, "seconds": 0.0,
             "prompt_tokens": 0}
    raw = []
    t0 = time.time()
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        ctx = [msgs[j] for j in range(max(0, b[0] - ctx_n), b[0])]
        lines = ["Context (do not extract from these):"] + [f"  [{m['date']}] {m['a']}: {m['t']}" for m in ctx]
        lines += ["NEW messages:"] + [f"  {k}. [{msgs[j]['date']}] {msgs[j]['a']}: {msgs[j]['t']}" for k, j in enumerate(b)]
        known_s = ", ".join(sorted(fs.entities())[-40:])
        user = (f"Projects already in memory: {known_s or 'none yet'}\n\n" + "\n".join(lines))
        text, pt, _ = llm(url, [{"role": "system", "content": EXTRACT_SYS},
                                {"role": "user", "content": user}], 60 + 40 * len(b), grammar=fact_grammar(len(b)))
        stats["calls"] += 1
        stats["prompt_tokens"] += pt or 0
        seen_text = " ".join(m["t"] for m in ctx).lower() + " " + " ".join(msgs[j]["t"] for j in b).lower()
        for j, o in parse_facts(text, b):
            name = str(o.get("project", ""))
            # an entity must be named somewhere in what the extractor saw, or already be in memory
            if name.lower() not in seen_text and name.lower() not in fs.names:
                stats["dropped_unseen_entity"] = stats.get("dropped_unseen_entity", 0) + 1
                continue
            attr = str(o.get("attr", "")).lower().replace("collaborators", "collaborator")
            val = o.get("value")
            if attr == "started":
                val = msgs[j]["date"]
            raw.append([j, name, attr, str(o.get("op", "set")).lower(), str(val)])
            fs.apply(name, attr, str(o.get("op", "set")).lower(), val, msgs[j]["date"], j)
            stats["facts"] += 1
        if log and stats["calls"] % 10 == 0:
            print(f"  extract {s + len(b)}/{len(idx)} facts={stats['facts']} {time.time() - t0:.0f}s", file=log, flush=True)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["raw"] = raw
    return fs, stats


# ---------------- memory hygiene ----------------
def _lev(a, b):
    """Optimal string alignment distance: an adjacent transposition costs 1 ('MaraiDB' ~ 'MariaDB')."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def ground(value, text):
    """A stated value must appear in the message. Snap near-misses to the message's own spelling
    ('MaraiDB' -> 'MariaDB', 'quokka_box' -> 'quokka'); return None when nothing in the text supports it.
    Snapping allows 1 edit below 8 characters and 2 from 8 up: 2 edits on short words turned 'host' into
    'Rust' and a filler word into the language 'best'."""
    v = value.strip()
    if re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", text, re.I):
        return v
    toks = re.findall(r"[A-Za-z0-9+#]+(?:\.[A-Za-z0-9]+)?", text)
    sub = [t for t in toks if len(t) >= 4 and t.lower() in v.lower()]
    if sub:
        return max(sub, key=len)
    lim = 1 if len(v) < 8 else 2
    near = [t for t in toks if len(t) >= 4 and _lev(t.lower(), v.lower()) <= lim]
    if near:
        return min(near, key=lambda t: _lev(t.lower(), v.lower()))
    return None


def recall_candidates(msgs, raw, fs, self_names=("daniel",), use_gate=False):
    """Messages worth a second, closer read: they produced no facts at all, yet either name a known
    project (or are the same author's follow-up to one that does), or are the user's own messages that
    passed the gate. In noisy batches the extractor skips or misaligns exactly these."""
    # judge by what the store accepted, not by what was extracted: a line full of a neighbour's facts
    # (later realigned or dropped) means this message's own facts were never read
    have = {src for h in fs.h.values() for _, _, src in h if src is not None}
    have |= {src for h in fs.collab.values() for _, _, _, src in h if src is not None}
    ents = [e.lower() for e in fs.entities()]
    out = []
    for j, m in enumerate(msgs):
        if j in have:
            continue
        low = m["t"].lower()
        named = any(_names(e, low) for e in ents)
        follow = (j > 0 and msgs[j - 1]["a"] == m["a"] and re.search(r"\bit\b", low)
                  and any(_names(e, msgs[j - 1]["t"]) for e in ents))
        own = m["a"].lower() in self_names and (not use_gate or GATE.search(m["t"]))
        if named or follow or own:
            out.append(j)
    return out


def recall_pass(msgs, raw, fs, url, self_names=("daniel",), use_gate=False, batch_size=3, log=None):
    """Re-read the recall candidates in small batches (3 messages, with context) and return extra raw facts."""
    cand = recall_candidates(msgs, raw, fs, self_names, use_gate)
    extra, t0 = [], time.time()
    stats = {"recall_candidates": len(cand), "recall_calls": 0}
    for s in range(0, len(cand), batch_size):
        b = cand[s:s + batch_size]
        ctx = [msgs[j] for j in range(max(0, b[0] - 3), b[0])]
        lines = ["Context (do not extract from these):"] + [f"  [{m['date']}] {m['a']}: {m['t']}" for m in ctx]
        lines += ["NEW messages:"] + [f"  {k}. [{msgs[j]['date']}] {msgs[j]['a']}: {msgs[j]['t']}" for k, j in enumerate(b)]
        user = f"Projects already in memory: {', '.join(fs.entities()) or 'none yet'}\n\n" + "\n".join(lines)
        text, _, _ = llm(url, [{"role": "system", "content": EXTRACT_SYS}, {"role": "user", "content": user}],
                         60 + 40 * len(b), grammar=fact_grammar(len(b)))
        stats["recall_calls"] += 1
        for j, o in parse_facts(text, b):
            attr = o["attr"]
            val = msgs[j]["date"] if attr == "started" else o["value"]
            extra.append([j, o["project"], attr, o["op"], str(val)])
        if log and stats["recall_calls"] % 20 == 0:
            print(f"  recall {s + len(b)}/{len(cand)} extra={len(extra)} {time.time() - t0:.0f}s", file=log, flush=True)
    stats["recall_facts"] = len(extra)
    stats["recall_seconds"] = round(time.time() - t0, 1)
    return extra, stats


NOT_NAMES = {"it", "this", "that", "they", "them", "project", "projects", "host", "server", "box", "repo", "app",
             "thing", "stuff", "the", "my", "our", "your", "one", "something", "everything", "none"}
TYPED = ("language", "database", "host")


def _names(name, text):
    return re.search(r"(?<![\w])" + re.escape(name.lower()) + r"(?![\w])", text.lower()) is not None


def _prepass(raw, msgs, st):
    """Batch-level repairs before facts are applied.
      - pronouns and filler words are never entity names ('it' became an entity on the 600K log)
      - value typing: a value seen mostly as one attribute (basalt: host) is rejected as another (database)
      - realignment: in noisy batches the extractor sometimes writes a message's facts on a neighbouring
        line. A fact whose value is not in its message is moved to the nearest message (within 12) that
        contains the value and names the project (or is the same author's follow-up to one that does).
    """
    from collections import Counter, defaultdict
    raw = [r for r in raw if r[1].strip().lower() not in NOT_NAMES and len(r[1].strip()) >= 3]
    typ = defaultdict(Counter)
    for j, name, attr, op, val in raw:
        if attr in TYPED and ground(str(val), msgs[j]["t"]):
            typ[str(val).lower()][attr] += 1
    out = []
    for j, name, attr, op, val in raw:
        val = str(val)
        if attr in TYPED and val.lower() in typ:
            c = typ[val.lower()]
            top, n = c.most_common(1)[0]
            if top != attr and n >= 2 and n >= 2 * c.get(attr, 0):
                st["mistyped_dropped"] = st.get("mistyped_dropped", 0) + 1
                continue
        if attr in TYPED + ("port", "collaborator") and not ground(val, msgs[j]["t"]):
            best = None
            for d in range(1, 13):
                for k in (j - d, j + d):
                    if not (0 <= k < len(msgs)) or not ground(val, msgs[k]["t"]):
                        continue
                    lead = k > 0 and msgs[k - 1]["a"] == msgs[k]["a"] and _names(name, msgs[k - 1]["t"])
                    if _names(name, msgs[k]["t"]) or lead:
                        best = k
                        break
                if best is not None:
                    break
            if best is not None:
                st["realigned"] = st.get("realigned", 0) + 1
                j = best
        out.append([j, name, attr, op, val])
    return sorted(out, key=lambda r: r[0])


def consolidate(raw, msgs, self_names=("daniel",), hygiene=True, stats=None):
    """Build a FactStore from raw extracted facts [msg_idx, project, attr, op, value].

    Hygiene rules (generic, not tied to the test templates):
      1. a project starts once: only the first 'started' counts
      2. the user is never their own collaborator
      3. grounding: language/database/host/port/collaborator values must appear in the message
         (near-misses snap to the message's spelling; unsupported values are dropped)
      4. coreference: a fact from a message that does not name its project is re-attached to the
         project the same author last named (within 6 messages); a message naming exactly one known
         project re-attaches facts that name a different, unmentioned project
    """
    st = stats if stats is not None else {}
    fs = FactStore()
    known = set()
    pre = {}
    raw = sorted(raw, key=lambda r: r[0])
    if hygiene:
        raw = _prepass(raw, msgs, st)
    for j, name, attr, op, val in raw:
        known.add(name.lower())
        text = msgs[j]["t"]
        if hygiene:
            low = text.lower()
            named = [k for k in known if re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", low)]
            if name.lower() not in low:
                if len(named) == 1:
                    tgt = named[0]
                else:
                    tgt = None
                    for i in range(j - 1, max(-1, j - 7), -1):
                        if msgs[i]["a"] != msgs[j]["a"]:
                            continue
                        hit = [k for k in known if re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", msgs[i]["t"].lower())]
                        if hit:
                            tgt = hit[0] if len(hit) == 1 else None
                            break
                if tgt and tgt != name.lower():
                    name = fs.names.get(tgt, tgt[:1].upper() + tgt[1:])
                    st["reattached"] = st.get("reattached", 0) + 1
                    st.setdefault("reattach_log", []).append([j, msgs[j]["t"][:80], tgt])
            if attr == "started" and (fs.canon(name), "started") in fs.h:
                st["restart_dropped"] = st.get("restart_dropped", 0) + 1
                continue
            if attr == "started" and self_names and msgs[j]["a"].lower() not in self_names:
                # "my projects" are the ones the user introduced; other people's "started using Qwen3"
                # in the same channels is not the user starting a project
                st["other_author_start_dropped"] = st.get("other_author_start_dropped", 0) + 1
                continue
            if attr == "started" and not _names(name, text):
                # an introduction names the project; a 'started' on "gonna pause it" is a misread
                st["unnamed_start_dropped"] = st.get("unnamed_start_dropped", 0) + 1
                continue
            if attr == "collaborator" and val.lower() in self_names:
                st["self_collab_dropped"] = st.get("self_collab_dropped", 0) + 1
                continue
            if attr in ("language", "database", "host", "port", "collaborator"):
                # the value may be stated in an adjacent message by the same author ("Zephram stuff today" /
                # "and it's on Redis now instead of Postgres"); the fact keeps this message's position
                window = " ".join(msgs[i]["t"] for i in (j - 1, j, j + 1)
                                  if 0 <= i < len(msgs) and msgs[i]["a"] == msgs[j]["a"])
                g = ground(val, text) or ground(val, window)
                if g is None:
                    st["ungrounded_dropped"] = st.get("ungrounded_dropped", 0) + 1
                    continue
                if g != val:
                    st["snapped"] = st.get("snapped", 0) + 1
                val = g
                # "moved X off kestrel onto ironbark" restates the old value; when the same message also
                # gives a different value for this attribute, the restated current value is not a change
                if attr != "collaborator":
                    key = (name.lower(), attr, j)
                    if key not in pre:  # value before this message touched it
                        pre[key] = fs.value(fs.canon(name), attr)
                    cur = pre[key]
                    others = [r for r in raw if r[0] == j and r[2] == attr and r[1].lower() == name.lower()
                              and str(r[4]).lower() != str(val).lower()]
                    if cur is not None and str(cur).lower() == str(val).lower() and others:
                        st["restated_dropped"] = st.get("restated_dropped", 0) + 1
                        continue
        fs.apply(name, attr, op, val, msgs[j]["date"], j)
    return fs


def store_accuracy(fs, oracle):
    """Compare final state and histories of an extracted store with the oracle store."""
    tot = ok = 0
    hist_tot = hist_ok = 0
    for e in oracle.entities():
        for a in ("started", "language", "database", "host", "port", "status"):
            g = oracle.value(e, a)
            if g is None:
                continue
            tot += 1
            ok += str(fs.value(e, a) or "").lower() == str(g).lower()
            gh = [str(v).lower() for _, v in oracle.history(e, a)]
            hh = [str(v).lower() for _, v in fs.history(e, a)]
            hist_tot += 1
            hist_ok += gh == hh
        tot += 1
        ok += sorted(map(str.lower, fs.collaborators(e))) == sorted(map(str.lower, oracle.collaborators(e)))
    extra = [e for e in fs.entities() if e.lower() not in {x.lower() for x in oracle.entities()}]
    return {"current_value_acc": round(ok / tot, 3), "history_exact": round(hist_ok / hist_tot, 3),
            "n_values": tot, "spurious_entities": extra}
