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

    def entities(self):
        return sorted({e for e, _ in self.h} | set(self.collab))

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
    "status (active, paused, shipped or abandoned), collaborator (write 'collaborator+=<person>' when someone "
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
            fs.apply(str(o.get("project", "")), attr, str(o.get("op", "set")).lower(), val, msgs[j]["date"], j)
            stats["facts"] += 1
        if log and stats["calls"] % 10 == 0:
            print(f"  extract {s + len(b)}/{len(idx)} facts={stats['facts']} {time.time() - t0:.0f}s", file=log, flush=True)
    stats["seconds"] = round(time.time() - t0, 1)
    return fs, stats


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
