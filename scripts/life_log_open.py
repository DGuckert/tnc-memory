"""Open-ended facts for the life log: things a typed project schema cannot hold.

Adds, on top of life_log.py's project world (unchanged, same seed):
  why         reasons attached to project changes   "Why did I move Ferrow off kestrel?"
  person      facts about people, some changing     "Where does mosswren live now?" / "...before?"
              (some stated by the person themselves, in first person: 'just moved to Napier!')
  preference  the user's own preferences, changing  "What editor do I use now?"
  plan        plans and deadlines, some revised     "When do I want Calyx demo-ready by?"
  incident    one-off hardware/life events          "What did I replace kestrel's PSU with?"

Gold answers are keyword groups: every group must match (any alternative in a group counts).

  python life_log_open.py --filler 250 --out work/lifelog/small2
  python life_log_open.py --filler 30000 --out work/lifelog/big2
"""
import argparse, datetime as dt, json, os, random
import life_log as LL

CITIES = ["Dunedin", "Wellington", "Nelson", "Hamilton", "Napier", "Invercargill", "Timaru"]
COMPANIES = ["Rocket Lab", "Xero", "Halter", "Tait", "Trade Me", "Fisher & Paykel"]
ANIMALS = [("dog", "Biscuit"), ("cat", "Pixel"), ("ferret", "Noodle"), ("dog", "Kumara")]

REASONS = {  # (reason text, keyword groups that must appear in an answer)
    "database": [("the {o} backups kept corrupting", [["backup"], ["corrupt"]]),
                 ("I needed proper full-text search", [["full-text", "full text", "search"]]),
                 ("{o} was eating all the RAM", [["RAM", "memory"]]),
                 ("I wanted real transactions", [["transaction"]])],
    "host": [("{o}'s SSD started throwing SMART errors", [["SSD", "SMART", "disk", "drive"]]),
             ("{o} kept overheating", [["overheat", "heat", "hot", "thermal"]]),
             ("it needs a GPU now", [["GPU", "graphics"]]),
             ("the power bill on {o} was silly", [["power", "electricity"], ["bill", "cost"]])],
    "port": [("a game server grabbed the old port", [["game"]]),
             ("the firewall only allows that range", [["firewall"]]),
             ("it clashed with Grafana", [["Grafana"]])],
    "paused": [("exams are coming up", [["exam"]]),
               ("I'm burned out on it", [["burn", "burnt", "burned", "tired"]]),
               ("I'm waiting on an upstream fix", [["upstream"]])],
    "abandoned": [("someone shipped the same thing first", [["someone", "already", "first", "beat"]]),
                  ("the API it depended on got shut down", [["API"]])],
    "language": [("the {o} build times were killing me", [["build"]]),
                 ("I needed better async", [["async"]])],
}
REASON_FMT = ["mostly because {r}", "(reason: {r})", "did that because {r}", "had to, {r}"]

PREFS = [  # (slot, question noun, [values in order], keyword for the slot)
    ("editor", "editor", ["Neovim", "Helix"]),
    ("laptop distro", "Linux distro on my laptop", ["Arch", "Debian"]),
    ("coffee order", "coffee order", ["flat white", "long black"]),
    ("keyboard", "keyboard", ["Corne", "Moonlander", "Corne"]),
]
PREF_FIRST = ["my {s} of choice is {v}", "I'm on {v} for my {s}", "{v} is my {s} these days"]
PREF_CHANGE = ["switched my {s} from {o} to {v}", "ditched {o}, my {s} is {v} now", "{s}: {v} now instead of {o}"]

INCIDENTS = [  # (date offset in days from START, text, question, keyword groups)
    (130, "kestrel's PSU died, swapped in a Corsair RM750", "What did I replace kestrel's PSU with?",
     [["Corsair", "RM750"]]),
    (175, "quokka got hit by a power cut and lost its RAID array", "What happened to quokka's RAID array?",
     [["power"], ["lost", "died", "gone", "failed", "destroyed", "wiped"]]),
    (205, "basalt got a new 4TB NVMe today", "What new hardware did basalt get?", [["4TB", "4 TB"], ["NVMe", "SSD"]]),
    (60, "spilled coffee on the wharf keyboard, it's toast", "What happened to the wharf keyboard?",
     [["coffee"]]),
    (240, "ironbark's fan started grinding so I swapped it for a Noctua", "What fan did ironbark get?",
     [["Noctua"]]),
]
PLANS = [  # (date offset, project, text, question, keyword groups, revision: (offset, text, groups) or None)
    (120, "Calyx", "want Calyx demo-ready by 15 September for the Christchurch hackathon",
     "When do I want Calyx demo-ready by?", [["15", "fifteenth"], ["Sep"]],
     (200, "pushed the Calyx demo to 29 September, hackathon moved", [["29"], ["Sep"]])),
    (110, "Calyx", None, "What is the Calyx demo deadline for?", [["hackathon"]], None),
    (150, "Brakka", "aiming to ship Brakka v1 before the end of July", "When am I aiming to ship Brakka v1?",
     [["July", "Jul"]], None),
    (190, "Kestova", "promised tibbet a Kestova beta by 20 August", "Who did I promise a Kestova beta to?",
     [["tibbet"]], None),
    (230, "wharf", "plan: get everything off wharf by October, it's dying", "By when do I plan to get everything off wharf?",
     [["Oct"]], None),
]


def make_open(events, seed=7):
    """Returns (blocks, open_events, questions). blocks: (date, [msg], tag) like LL.render's."""
    rng = random.Random(seed + 10)
    S = LL.START
    blocks, oev, qs = [], [], []

    def add(d, msgs, ev):
        oev.append(ev)
        blocks.append((d, msgs, ("o", len(oev) - 1)))

    def dm(text, author="daniel", ch=None):
        return {"ch": ch or rng.choice(["general", "dev-log", "random"]), "a": author, "t": text}

    # --- reasons attached to project changes ---
    changes = [(i, e) for i, e in enumerate(events) if (e[5] is not None and e[2] in ("database", "host", "port", "language"))
               or (e[2] == "status" and e[4] in ("paused", "abandoned"))]
    rng.shuffle(changes)
    used = set()
    for i, (d, p, a, op, v, o, _) in changes:
        if len([q for q in qs if q["type"] == "why"]) >= 12 or (p, a) in used:
            continue
        used.add((p, a))
        key = v if a == "status" else a
        r, groups = rng.choice(REASONS[key])
        r = r.format(o=o, p=p)
        add(d, [dm(rng.choice(REASON_FMT).format(r=r))], {"type": "reason", "subject": p, "about": [a, v, o], "text": r})
        action = {"database": f"move {p} off {o}", "host": f"move {p} off {o}", "port": f"change {p}'s port from {o}",
                  "language": f"rewrite {p} out of {o}", "paused": f"pause {p}", "abandoned": f"abandon {p}"}[key]
        qs.append({"type": "why", "q": f"Why did I {action}?", "gold": groups, "kind": "keywords"})

    # --- people ---
    for k, person in enumerate(LL.PEOPLE):
        city = rng.choice(CITIES)
        d0 = S + dt.timedelta(days=rng.randint(5, 60))
        self_said = rng.random() < 0.5
        add(d0, [dm(f"I'm based in {city}" if self_said else f"{person} is based in {city}", person if self_said else "daniel")],
            {"type": "person", "subject": person, "rel": "lives in", "value": city})
        moved = k % 2 == 0
        if moved:
            new = rng.choice([c for c in CITIES if c != city])
            d1 = d0 + dt.timedelta(days=rng.randint(60, 200))
            txt = (f"just moved to {new}! bye {city}" if rng.random() < 0.5 else f"finally out of {city}, living in {new} now")
            add(d1, [dm(txt, person)], {"type": "person", "subject": person, "rel": "lives in", "value": new, "old": city})
            qs.append({"type": "person", "q": f"Where does {person} live now?", "gold": [[new]], "kind": "keywords"})
            qs.append({"type": "person", "q": f"Where did {person} live before {new}?", "gold": [[city]], "kind": "keywords"})
        co = rng.choice(COMPANIES)
        d2 = S + dt.timedelta(days=rng.randint(10, 120))
        add(d2, [dm(f"{person} works at {co}")], {"type": "person", "subject": person, "rel": "works at", "value": co})
        if k % 3 == 1:
            new_co = rng.choice([c for c in COMPANIES if c != co])
            d3 = d2 + dt.timedelta(days=rng.randint(40, 150))
            add(d3, [dm(f"{person} quit {co} and joined {new_co}")],
                {"type": "person", "subject": person, "rel": "works at", "value": new_co, "old": co})
            qs.append({"type": "person", "q": f"Where does {person} work now?", "gold": [[new_co.split()[0]]], "kind": "keywords"})
        if k < len(ANIMALS):
            an, nm = ANIMALS[k]
            d4 = S + dt.timedelta(days=rng.randint(30, 250))
            add(d4, [dm(f"{person} got a {an} called {nm}")], {"type": "person", "subject": person, "rel": "pet", "value": f"{an} {nm}"})
            qs.append({"type": "person", "q": f"What is the name of {person}'s {an}?", "gold": [[nm]], "kind": "keywords"})

    # --- the user's preferences ---
    for slot, noun, vals in PREFS:
        d = S + dt.timedelta(days=rng.randint(3, 40))
        add(d, [dm(rng.choice(PREF_FIRST).format(s=slot, v=vals[0]))], {"type": "pref", "subject": "me", "rel": slot, "value": vals[0]})
        for o, v in zip(vals, vals[1:]):
            d = d + dt.timedelta(days=rng.randint(50, 110))
            add(d, [dm(rng.choice(PREF_CHANGE).format(s=slot, o=o, v=v))],
                {"type": "pref", "subject": "me", "rel": slot, "value": v, "old": o})
        qs.append({"type": "preference", "q": f"What {noun} do I use now?" if slot != "coffee order" else "What's my coffee order now?",
                   "gold": [[vals[-1]]], "kind": "keywords"})
        qs.append({"type": "preference", "q": f"What {noun} did I use before {vals[-1]}?" if slot != "coffee order"
                   else f"What did I order before switching to {vals[-1]}?", "gold": [[vals[-2]]], "kind": "keywords"})

    # --- plans ---
    for off, subj, text, q, groups, rev in PLANS:
        if text:
            add(S + dt.timedelta(days=off), [dm(text)], {"type": "plan", "subject": subj, "text": text})
        if rev:
            add(S + dt.timedelta(days=rev[0]), [dm(rev[1])], {"type": "plan", "subject": subj, "text": rev[1], "revises": text})
            groups = rev[2]
        qs.append({"type": "plan", "q": q, "gold": groups, "kind": "keywords"})

    # --- incidents ---
    for off, text, q, groups in INCIDENTS:
        add(S + dt.timedelta(days=off), [dm(text)], {"type": "incident", "subject": text.split("'")[0].split()[0], "text": text})
        qs.append({"type": "incident", "q": q, "gold": groups, "kind": "keywords"})

    return blocks, oev, qs


def build(filler_n, seed=7):
    events = LL.make_world(seed)
    blocks = [(d, ms, ("e", eids[0])) for d, ms, eids in LL.render(events, seed)]
    ob, oev, oqs = make_open(events, seed)
    # an open event that explains a project change goes right after that change's messages
    blocks = sorted(blocks + ob, key=lambda b: (b[0], 0 if b[2][0] == "e" else 1))
    rng = random.Random(seed + 3)
    filler = []
    if filler_n:
        with open("/home/daniel/tnc/corpus.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
        while len(filler) < filler_n:
            s = rng.randrange(0, len(lines) - 40)
            run = [json.loads(l) for l in lines[s:s + rng.randint(5, 40)]]
            filler.extend({"ch": m["ch"], "a": m["a"], "t": m["t"][:400]} for m in run if m["t"].strip())
        filler = filler[:filler_n]
    days = (LL.END - LL.START).days
    fdates = sorted(LL.START + dt.timedelta(days=int(i * days / max(1, len(filler)))) for i in range(len(filler)))
    msgs, bi, fi = [], 0, 0
    while bi < len(blocks) or fi < len(filler):
        if bi < len(blocks) and (fi >= len(filler) or blocks[bi][0] <= fdates[fi]):
            d, ms, (kind, idx) = blocks[bi]
            for m in ms:
                msgs.append(dict(m, date=LL.fmt_date(d), ev=idx if kind == "e" else None, oev=idx if kind == "o" else None))
            bi += 1
        else:
            msgs.append(dict(filler[fi], date=LL.fmt_date(fdates[fi]), ev=None, oev=None)); fi += 1
    return events, oev, oqs, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filler", type=int, default=250)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    events, oev, oqs, msgs = build(a.filler, a.seed)
    qs = LL.make_questions(events, a.seed)
    for q in oqs:
        q["id"] = len(qs)
        qs.append(q)
    with open(os.path.join(a.out, "log.jsonl"), "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m) + "\n")
    json.dump([[str(e[0]), e[1], e[2], e[3], e[4], e[5], e[6]] for e in events], open(os.path.join(a.out, "events.json"), "w"), indent=0)
    json.dump(oev, open(os.path.join(a.out, "open_events.json"), "w"), indent=1)
    json.dump(qs, open(os.path.join(a.out, "questions.json"), "w"), indent=1)
    from collections import Counter
    print(len(events), "project events,", len(oev), "open events,", len(msgs), "msgs;", dict(Counter(q["type"] for q in qs)))


if __name__ == "__main__":
    main()
