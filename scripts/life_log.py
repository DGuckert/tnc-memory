"""Synthetic 'life log' for testing memory that behaves like context, not just lookup.

A 10-month timeline (2026-01-05 .. 2026-10-30) of invented projects whose attributes CHANGE:
ports move, databases migrate, hosts change, status flips, collaborators join and leave.
Event messages are written in varied casual phrasings (some only say "it"), then buried in
real Discord chatter from corpus.jsonl (filler stays local, never committed).

Question types (gold computed from the event log, never from text):
  current     value now, after it changed at least once          ("what port is X on now?")
  past        value at an earlier time / before a change         ("what host was X on in March?")
  history     every value in order                               ("list every db X has used")
  aggregate   set of entities matching a condition now           ("which projects use Postgres?")
  count       how many entities match                            ("how many projects are paused?")
  multi_hop   find entity by one fact, answer another            ("db of the project on port 8443?")
  temporal    order of events                                    ("which did I start first, A or B?")

  python life_log.py --filler 250 --out work/lifelog/small     (fits in an 8K context window)
  python life_log.py --filler 30000 --out work/lifelog/big     (~1M tokens: only memory can cope)
Both variants use the same seed, so the same projects, events and questions.
"""
import argparse, datetime as dt, json, os, random

PROJECTS = ["Ferrow", "Quillon", "Brakka", "Venmire", "Ostrel", "Calyx", "Tundrel", "Mirevale",
            "Pexwick", "Sorrel", "Halvane", "Drisk", "Umbrin", "Kestova", "Lorrish", "Zephram"]
HOSTS = ["kestrel", "basalt", "quokka", "ironbark", "pumice", "wharf"]
PEOPLE = ["mosswren", "tibbet", "juno_k", "farrowdale", "pickle_rick77", "veyra"]
LANGS = ["Rust", "Go", "Python", "TypeScript", "Kotlin", "C++"]
DBS = ["Postgres", "SQLite", "Redis", "MongoDB", "MariaDB"]
STATUSES = ["active", "paused", "shipped", "abandoned"]
START, END = dt.date(2026, 1, 5), dt.date(2026, 10, 30)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]

T = {  # phrasings; {p}=project {v}=value {o}=old value
    "start": ["starting a new side project called {p}", "ok new thing: {p}. no idea if it'll go anywhere",
              "spun up a repo for {p} tonight", "kicked off {p} today, been meaning to for weeks"],
    "language": ["{p} is going to be written in {v}", "doing {p} in {v} this time",
                 "decided on {v} for {p}", "{p} -> {v}, fight me"],
    "language_change": ["rewrote {p} from {o} to {v}, way nicer now", "{p} is {v} now, ripped out all the {o}"],
    "database": ["{p} uses {v} for storage", "went with {v} as the db for {p}", "{p} talks to a {v} instance"],
    "database_change": ["migrated {p} off {o} onto {v}", "{p} finally moved from {o} to {v}",
                        "swapped the {o} out of {p}, it's on {v} now"],
    "host": ["{p} lives on {v}", "deployed {p} to {v}", "{p} is running on the {v} box"],
    "host_change": ["moved {p} from {o} to {v}", "{p} got evicted off {o}, it's on {v} now",
                    "migrated {p} over to {v}, {o} was too full"],
    "port": ["{p} listens on port {v}", "{p} is on :{v}", "put {p} on port {v}"],
    "port_change": ["changed {p}'s port from {o} to {v}", "{p} clashed with something on {o}, moved it to {v}",
                    "{p} is on :{v} now (was {o})"],
    "status_paused": ["putting {p} on ice for a bit", "pausing {p}, no time", "{p} is on hold"],
    "status_active": ["back on {p} again", "picked {p} back up", "resuming {p}"],
    "status_shipped": ["{p} is shipped!! v1 is out", "shipped {p} finally", "{p} is done and released"],
    "status_abandoned": ["killing {p}, it was a dead end", "{p} is dead, not worth it", "abandoning {p}"],
    "collab_add": ["{v} is helping out on {p} now", "{v} joined {p}", "roped {v} into working on {p}"],
    "collab_remove": ["{v} dropped off {p}", "{v} isn't on {p} anymore", "{v} left {p} to do other stuff"],
}
PRONOUN = {  # second message of a pair that only says "it"; first message names the project
    "port_change": "moved it from :{o} to :{v}", "host_change": "shifted it off {o} onto {v}",
    "database_change": "and it's on {v} now instead of {o}", "status_paused": "gonna pause it for now",
}
LEADS = ["been poking at {p} again", "ok {p} update", "{p} stuff today", "looking at {p}"]


def fmt_date(d):
    return d.isoformat()


def make_world(seed=7):
    rng = random.Random(seed)
    days = (END - START).days
    starts = sorted(rng.sample(range(0, int(days * 0.55)), len(PROJECTS)))
    ports = rng.sample(range(3000, 9900), 60)
    events = []  # (date, project, attr, op, value, old, author)
    state = {}
    for p, sd in zip(PROJECTS, starts):
        d = START + dt.timedelta(days=sd)
        s = {"language": rng.choice(LANGS), "database": rng.choice(DBS), "host": rng.choice(HOSTS),
             "port": ports.pop(), "status": "active", "collab": set()}
        events.append((d, p, "started", "set", fmt_date(d), None, "daniel"))
        for a in ("language", "database", "host", "port"):
            events.append((d + dt.timedelta(days=rng.randint(0, 3)), p, a, "set", s[a], None, "daniel"))
        if rng.random() < 0.6:
            c = rng.choice(PEOPLE); s["collab"].add(c)
            events.append((d + dt.timedelta(days=rng.randint(2, 10)), p, "collab", "add", c, None, "daniel"))
        # later changes, strictly after the setup days
        t = d + dt.timedelta(days=12)
        for _ in range(rng.randint(2, 5)):
            t = t + dt.timedelta(days=rng.randint(8, 40))
            if t >= END:
                break
            kind = rng.choices(["port", "database", "host", "status", "collab", "language"],
                               [3, 2, 3, 2, 2, 0.5])[0]
            if s["status"] in ("shipped", "abandoned") and kind != "status":
                continue
            if kind == "port":
                o, v = s["port"], ports.pop(); s["port"] = v
                events.append((t, p, "port", "set", v, o, "daniel"))
            elif kind in ("database", "host", "language"):
                pool = {"database": DBS, "host": HOSTS, "language": LANGS}[kind]
                o = s[kind]; v = rng.choice([x for x in pool if x != o]); s[kind] = v
                events.append((t, p, kind, "set", v, o, "daniel"))
            elif kind == "status":
                if s["status"] == "active":
                    v = rng.choices(["paused", "shipped", "abandoned"], [3, 1.5, 1])[0]
                elif s["status"] == "paused":
                    v = rng.choice(["active", "abandoned"])
                else:
                    continue
                o = s["status"]; s["status"] = v
                events.append((t, p, "status", "set", v, o, "daniel"))
            else:
                if s["collab"] and rng.random() < 0.45:
                    c = rng.choice(sorted(s["collab"])); s["collab"].discard(c)
                    events.append((t, p, "collab", "remove", c, None, "daniel"))
                else:
                    c = rng.choice([x for x in PEOPLE if x not in s["collab"]]); s["collab"].add(c)
                    events.append((t, p, "collab", "add", c, None, rng.choice(["daniel", c])))
        state[p] = s
    events.sort(key=lambda e: (e[0], PROJECTS.index(e[1])))
    return events


def render(events, seed=7):
    """Event list -> list of message blocks [(date, [msg, ...], [event_ids])]."""
    rng = random.Random(seed + 1)
    blocks = []
    for i, (d, p, a, op, v, o, author) in enumerate(events):
        if a == "started":
            key = "start"
        elif a in ("language", "database", "host", "port"):
            key = a + ("_change" if o is not None else "")
        elif a == "status":
            key = "status_" + v
        else:
            key = "collab_" + op
        pron = key in PRONOUN and rng.random() < 0.4
        if pron:
            msgs = [rng.choice(LEADS).format(p=p), PRONOUN[key].format(p=p, v=v, o=o)]
        else:
            msgs = [rng.choice(T[key]).format(p=p, v=v, o=o)]
        ch = rng.choice(["projects", "dev-log", "general"])
        blocks.append((d, [{"ch": ch, "a": author if a == "collab" else "daniel", "t": m} for m in msgs], [i]))
    return blocks


def state_at(events, when=None):
    """Replay events up to and including date `when` (None = all)."""
    st = {}
    for d, p, a, op, v, o, _ in events:
        if when and d > when:
            break
        s = st.setdefault(p, {"collab": set(), "status": "active"})
        if a == "collab":
            (s["collab"].add if op == "add" else s["collab"].discard)(v)
        else:
            s[a] = v
    return st


def history(events, p, a):
    out = ["active"] if a == "status" else []  # every project starts active (no explicit event)
    for d, pp, aa, op, v, o, _ in events:
        if pp == p and aa == a and (not out or out[-1] != v):
            out.append(v)
    return out


def make_questions(events, seed=7, per_type=12):
    rng = random.Random(seed + 2)
    now = state_at(events)
    qs = []
    changed = [(p, a) for p in PROJECTS for a in ("port", "database", "host", "status", "language")
               if len(history(events, p, a)) >= 2]
    names = {"port": "port", "database": "database", "host": "host machine", "status": "status",
             "language": "language"}
    # current
    for p, a in rng.sample(changed, min(per_type, len(changed))):
        qs.append({"type": "current", "q": f"What {names[a]} is {p} on right now?" if a in ("port", "host")
                   else f"What {names[a]} does {p} use now?" if a != "status" else f"What's the status of {p} now?",
                   "gold": str(now[p][a]), "kind": "value"})
    # past: value before the most recent change
    for p, a in rng.sample(changed, min(per_type, len(changed))):
        h = history(events, p, a)
        qs.append({"type": "past", "q": f"Before {p}'s {names[a]} was {h[-1]}, what was it?",
                   "gold": str(h[-2]), "kind": "value"})
    # history
    long_ = [(p, a) for p, a in changed if len(history(events, p, a)) >= 3]
    short = [(p, a) for p, a in changed if len(history(events, p, a)) == 2 and a != "status"]
    multi = rng.sample(long_, min(per_type, len(long_)))
    multi += rng.sample(short, min(per_type - len(multi), len(short)))
    for p, a in multi:
        h = [str(x) for x in history(events, p, a)]
        qs.append({"type": "history", "q": f"List every {names[a]} {p} has had, oldest first.",
                   "gold": h, "kind": "sequence"})
    # aggregate (current)
    agg = []
    for db in DBS:
        s = sorted(p for p in PROJECTS if now[p].get("database") == db)
        if len(s) >= 2:
            agg.append((f"Which projects use {db} right now?", s))
    for h_ in HOSTS:
        s = sorted(p for p in PROJECTS if now[p].get("host") == h_ and now[p]["status"] in ("active", "paused"))
        if len(s) >= 2:
            agg.append((f"Which active or paused projects are running on {h_}?", s))
    for c in PEOPLE:
        s = sorted(p for p in PROJECTS if c in now[p]["collab"])
        if len(s) >= 2:
            agg.append((f"Which projects is {c} working on right now?", s))
    for st_ in ("paused", "shipped", "active"):
        s = sorted(p for p in PROJECTS if now[p]["status"] == st_)
        if 2 <= len(s) <= 8:
            agg.append((f"Which projects are {st_} right now?", s))
    for q, s in rng.sample(agg, min(per_type, len(agg))):
        qs.append({"type": "aggregate", "q": q, "gold": s, "kind": "set"})
    # count
    cnt = []
    for st_ in STATUSES:
        cnt.append((f"How many of my projects are {st_} right now?", sum(now[p]["status"] == st_ for p in PROJECTS)))
    for lang in LANGS:
        cnt.append((f"How many projects are written in {lang} at the moment?", sum(now[p]["language"] == lang for p in PROJECTS)))
    for i, m in enumerate(MONTHS[:6]):
        n = sum(1 for e in events if e[2] == "started" and e[0].month == i + 1)
        cnt.append((f"How many new projects did I start in {m}?", n))
    for q, n in rng.sample(cnt, min(per_type, len(cnt))):
        qs.append({"type": "count", "q": q, "gold": n, "kind": "count"})
    # multi-hop
    mh = []
    for p in PROJECTS:
        if now[p]["status"] not in ("shipped", "abandoned"):
            mh.append((f"What database does the project on port {now[p]['port']} use?", now[p]["database"]))
            mh.append((f"What language is the project currently on port {now[p]['port']} written in?", now[p]["language"]))
    from collections import Counter
    moves = Counter((v, d.month) for d, p, a, op, v, o, _ in events if a == "host" and o is not None)
    for d, p, a, op, v, o, _ in events:
        if a == "host" and o is not None and now[p]["collab"] and moves[(v, d.month)] == 1:
            mh.append((f"Who is working with me on the project I moved to {v} in {MONTHS[d.month - 1]}?",
                       sorted(now[p]["collab"])))
    rng.shuffle(mh)
    seen = set()
    for q, g in mh:
        if len([x for x in qs if x["type"] == "multi_hop"]) >= per_type:
            break
        if q in seen:
            continue
        seen.add(q)
        qs.append({"type": "multi_hop", "q": q, "gold": g, "kind": "set" if isinstance(g, list) else "value"})
    # temporal
    starts = {e[1]: e[0] for e in events if e[2] == "started"}
    order = sorted(PROJECTS, key=lambda p: starts[p])
    tp = []
    for _ in range(40):
        a, b = rng.sample(PROJECTS, 2)
        if abs((starts[a] - starts[b]).days) >= 5:
            tp.append((f"Which did I start first, {a} or {b}?", min(a, b, key=lambda p: starts[p])))
    for i in range(len(order) - 1):
        if (starts[order[i + 1]] - starts[order[i]]).days >= 1:
            tp.append((f"What was the next project I started after {order[i]}?", order[i + 1]))
    rng.shuffle(tp)
    seen = set()
    for q, g in tp:
        if len([x for x in qs if x["type"] == "temporal"]) >= per_type:
            break
        if q not in seen:
            seen.add(q)
            qs.append({"type": "temporal", "q": q, "gold": g, "kind": "value"})
    for i, q in enumerate(qs):
        q["id"] = i
    return qs


def build(filler_n, seed=7, corpus="/home/daniel/tnc/corpus.jsonl"):
    events = make_world(seed)
    blocks = render(events, seed)
    rng = random.Random(seed + 3)
    # filler: contiguous runs of real chatter, so conversations stay coherent
    filler = []
    if filler_n:
        with open(corpus, encoding="utf-8") as f:
            lines = f.readlines()
        while len(filler) < filler_n:
            s = rng.randrange(0, len(lines) - 40)
            run = [json.loads(l) for l in lines[s:s + rng.randint(5, 40)]]
            filler.extend({"ch": m["ch"], "a": m["a"], "t": m["t"][:400]} for m in run if m["t"].strip())
        filler = filler[:filler_n]
    # spread filler over the timeline; event blocks go in at their dates
    days = (END - START).days
    fdates = sorted(START + dt.timedelta(days=int(i * days / max(1, len(filler)))) for i in range(len(filler)))
    msgs, gold_msgs = [], []
    bi = fi = 0
    while bi < len(blocks) or fi < len(filler):
        if bi < len(blocks) and (fi >= len(filler) or blocks[bi][0] <= fdates[fi]):
            d, ms, eids = blocks[bi]
            for m in ms:
                m = dict(m, date=fmt_date(d), ev=eids[0]); msgs.append(m)
            bi += 1
        else:
            msgs.append(dict(filler[fi], date=fmt_date(fdates[fi]), ev=None)); fi += 1
    return events, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filler", type=int, default=250)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    events, msgs = build(a.filler, a.seed)
    qs = make_questions(events, a.seed)
    with open(os.path.join(a.out, "log.jsonl"), "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m) + "\n")
    json.dump([[str(e[0]), e[1], e[2], e[3], e[4], e[5], e[6]] for e in events],
              open(os.path.join(a.out, "events.json"), "w"), indent=0)
    json.dump(qs, open(os.path.join(a.out, "questions.json"), "w"), indent=1)
    from collections import Counter
    print(len(events), "events,", sum(m["ev"] is not None for m in msgs), "event msgs,", len(msgs), "msgs total;",
          dict(Counter(q["type"] for q in qs)))


if __name__ == "__main__":
    main()
