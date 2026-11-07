"""Multi-project reference-resolution eval with real Discord chatter as noise.

Phrasings are deliberately disjoint from make_hard_eval.py (which stays the independent check).
Each project has two facts (different nouns), so the reader must get both project and noun right.

Cases
  explicit, typo              name in the question (typo = 1-2 edits)
  pronoun_single              one project, chatter, "its X?"
  pronoun_focus               A and B introduced, user asked about A, then "and its Y?"  -> A
  other_after_q               A and B introduced, asked about A, "the other one's Y?"     -> B
  other_two                   A then B introduced, no question yet, "the other one"       -> A (non-focus)
  chain_other                 asked A, "the other one?" (B), then "and its Y?"            -> B
  newest                      2-3 introduced, "the new one / latest one"                  -> last
  first                       2-3 introduced, "the first / original one"                  -> first
  second                      3 introduced, "the second one"                              -> second
  negation                    A and B, "not A, the other one" / "no not A"                -> B
  person_distractor           chatter names people who look like names ("Marta said ..."), 1 project
  clarify_other3              3 introduced, no prior question, "the other one"            -> clarify
  clarify_none                no project mentioned, "what's its X?"                       -> clarify
"""
import json, random, re, os

SEED = 4242
WORK = "/home/daniel/tnc/work"
NOUNS = ["access code", "invoice number", "serial number", "passphrase", "license key", "tracking ID"]
rng = random.Random(SEED)


def name():
    syl = ["ka", "vor", "lin", "tes", "mar", "quo", "bel", "dru", "zan", "fi", "rok", "sel", "tam", "vex", "nor",
           "pil", "gra", "hoth", "ul", "yen", "brim", "cas", "dov", "ess", "jor"]
    return "".join(rng.choice(syl) for _ in range(rng.choice([2, 2, 3]))).capitalize()


def code():
    a = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(a) for _ in range(3)) + "-" + "".join(rng.choice(a) for _ in range(4))


def typo(n):
    i = rng.randrange(1, len(n) - 1)
    op = rng.choice(["swap", "drop", "sub"])
    if op == "swap" and i < len(n) - 1:
        return n[:i] + n[i + 1] + n[i] + n[i + 2:]
    if op == "drop":
        return n[:i] + n[i + 1:]
    return n[:i] + rng.choice("aeiou") + n[i + 1:]


PEOPLE = ["Marta", "Kenji", "Dmitri", "Priya", "Tomasz", "Aroha", "Lucas", "Ingrid", "Wiremu", "Sofia"]
INTRO = ["started on {n} today", "{n} kicked off this morning", "oh and {n} is a thing now",
         "picked up {n} from the backlog", "client signed off on {n}", "{n} landed on my desk",
         "we're spinning up {n}", "got handed {n} as well"]
ACK = ["nice", "ok cool", "mhm", "noted", "sweet", "ah right", "k", "yep"]
PEOPLE_LINES = ["{p} said the build is green again", "{p} is out until monday", "ask {p}, they set it up",
                "{p} reckons we should wait", "{p} pinged me about lunch"]
Q_EXPL = ["remind me of the {noun} for {n}", "{n} {noun}?", "got the {noun} handy for {n}?",
          "need {n}'s {noun} real quick", "what was {n}'s {noun} again"]
Q_PRON = ["and its {noun}?", "what's the {noun} on that one", "remind me of its {noun}",
          "got the {noun} for that project?", "and the {noun} for it"]
Q_OTHER = ["and the {noun} for the other one?", "what about the other one's {noun}", "other project's {noun}?",
           "same for the other one, the {noun}"]
Q_NEW = ["what's the {noun} for the new one", "the latest one's {noun}?", "{noun} for the one that just came in",
         "and the newest project's {noun}"]
Q_FIRST = ["what's the {noun} for the first one", "the original project's {noun}?", "{noun} for the older one",
           "and the {noun} for the first project"]
Q_SECOND = ["what's the {noun} for the second one", "the second project's {noun}?"]
Q_NEG = ["not {n}, the other one - what's its {noun}", "no not {n}. the {noun} for the other one",
         "the {noun}, but not for {n}"]
Q_ASKED = ["what's the {noun} for {n}?", "{n}'s {noun} please"]
ANSWER = ["it's {c}", "{c}", "that'd be {c}"]


HELDOUT = {
    # written after the tracker was frozen; never used for tuning
    "INTRO": ["finally got the green light for {n}", "{n} came through from sales", "adding {n} to the pile",
              "heads up, {n} is ours now", "{n} got approved yesterday"],
    "Q_EXPL": ["{noun} for {n} pls", "do we still have {n}'s {noun}", "quick one, {n} {noun}"],
    "Q_PRON": ["what's its {noun} then", "{noun} for that?", "and that one's {noun}", "and for it, the {noun}"],
    "Q_OTHER": ["ok now the other one, {noun}", "and the {noun} of the other project", "flip to the other one - {noun}?"],
    "Q_NEW": ["{noun} for the most recent one", "the one we just added, {noun}?", "newer one's {noun}"],
    "Q_FIRST": ["{noun} for the one from earlier", "the earliest one's {noun}", "go back to the original, {noun}?"],
    "Q_SECOND": ["the 2nd one's {noun}", "{noun} for the middle one"],
    "Q_NEG": ["the {noun} - and I don't mean {n}, the other one", "forget {n}, what's the other one's {noun}",
              "{noun} for the one that isn't {n}"],
    "Q_ASKED": ["{n} {noun}?", "can I get {n}'s {noun}"],
}


def chatter_pool():
    pool = []
    for l in open("/home/daniel/tnc/corpus.jsonl", encoding="utf-8"):
        t = json.loads(l)["t"].strip().replace("\n", " ")
        if 20 <= len(t) <= 140 and "http" not in t and "<@" not in t and not re.search(
                r"\b(project|access code|invoice|serial|passphrase|license|tracking)\b", t, re.I):
            pool.append(t)
    rng.shuffle(pool)
    return pool[:20000]


def main():
    import sys
    global rng
    heldout = "--heldout" in sys.argv
    out_name = "reference_heldout" if heldout else "reference_eval"
    if heldout:
        rng = random.Random(SEED + 1)
        globals().update(HELDOUT)
    pool = chatter_pool()
    names = []
    while len(names) < 120:
        n = name()
        if n not in names and len(n) >= 5:
            names.append(n)
    projects = []
    for n in names:
        nouns = rng.sample(NOUNS, 2)
        projects.append({"name": n, "facts": {nn: code() for nn in nouns}})
    by = {p["name"]: p for p in projects}
    weights = {"explicit": 8, "typo": 8, "pronoun_single": 6, "pronoun_focus": 10, "other_after_q": 12,
               "other_two": 6, "chain_other": 6, "newest": 10, "first": 8, "second": 4, "negation": 6,
               "person_distractor": 6, "clarify_other3": 5, "clarify_none": 5}
    cases = list(weights)
    rows = []
    for i in range(400):
        case = rng.choices(cases, [weights[c] for c in cases])[0]
        msgs = []

        def chat(n=None):
            for _ in range(n if n is not None else rng.choice([1, 2, 3])):
                msgs.append({"role": "user", "content": rng.choice(pool)})
                msgs.append({"role": "assistant", "content": rng.choice(ACK)})

        def intro(p):
            msgs.append({"role": "user", "content": rng.choice(INTRO).format(n=p["name"])})
            msgs.append({"role": "assistant", "content": rng.choice(ACK)})

        def asked(p):
            nn = rng.choice(list(p["facts"]))
            msgs.append({"role": "user", "content": rng.choice(Q_ASKED).format(noun=nn, n=p["name"])})
            msgs.append({"role": "assistant", "content": rng.choice(ANSWER).format(c=p["facts"][nn])})

        k = 3 if case in ("second", "clarify_other3") else (rng.choice([2, 3]) if case in ("newest", "first") else 2)
        ps = rng.sample(projects, k)
        chat(rng.choice([1, 2, 3]))
        target = None
        if case in ("explicit", "typo", "pronoun_single", "person_distractor"):
            ps = ps[:rng.choice([1, 2])] if case in ("explicit", "typo") else ps[:1]
            for p in ps:
                intro(p); chat(rng.choice([0, 1, 2]))
            if case == "person_distractor":
                for _ in range(2):
                    msgs.append({"role": "user", "content": rng.choice(PEOPLE_LINES).format(p=rng.choice(PEOPLE))})
                    msgs.append({"role": "assistant", "content": rng.choice(ACK)})
            target = rng.choice(ps) if case in ("explicit", "typo") else ps[0]
            nn = rng.choice(list(target["facts"]))
            if case in ("explicit", "typo"):
                nm = typo(target["name"]) if case == "typo" else target["name"]
                q = rng.choice(Q_EXPL).format(noun=nn, n=nm)
            else:
                q = rng.choice(Q_PRON).format(noun=nn)
        elif case in ("pronoun_focus", "other_after_q", "chain_other"):
            a, b = ps[0], ps[1]
            order = [a, b] if rng.random() < 0.5 else [b, a]
            for p in order:
                intro(p); chat(rng.choice([0, 1]))
            asked(a); chat(rng.choice([0, 1, 2]))
            if case == "pronoun_focus":
                target = a
                nn = rng.choice(list(a["facts"])); q = rng.choice(Q_PRON).format(noun=nn)
            elif case == "other_after_q":
                target = b
                nn = rng.choice(list(b["facts"])); q = rng.choice(Q_OTHER).format(noun=nn)
            else:
                n1 = rng.choice(list(b["facts"]))
                msgs.append({"role": "user", "content": rng.choice(Q_OTHER).format(noun=n1)})
                msgs.append({"role": "assistant", "content": rng.choice(ANSWER).format(c=b["facts"][n1])})
                chat(rng.choice([0, 1]))
                target = b
                nn = rng.choice(list(b["facts"])); q = rng.choice(Q_PRON).format(noun=nn)
        elif case == "other_two":
            for p in ps:
                intro(p); chat(rng.choice([0, 1, 2]))
            target = ps[0]
            nn = rng.choice(list(target["facts"])); q = rng.choice(Q_OTHER).format(noun=nn)
        elif case in ("newest", "first", "second"):
            for p in ps:
                intro(p); chat(rng.choice([0, 1, 2]))
            target = {"newest": ps[-1], "first": ps[0], "second": ps[1] if len(ps) > 1 else ps[0]}[case]
            nn = rng.choice(list(target["facts"]))
            q = rng.choice({"newest": Q_NEW, "first": Q_FIRST, "second": Q_SECOND}[case]).format(noun=nn)
        elif case == "negation":
            for p in ps:
                intro(p); chat(rng.choice([0, 1]))
            excl = rng.choice(ps)
            target = [p for p in ps if p is not excl][0]
            nn = rng.choice(list(target["facts"]))
            q = rng.choice(Q_NEG).format(noun=nn, n=excl["name"])
        elif case == "clarify_other3":
            for p in ps:
                intro(p); chat(rng.choice([0, 1]))
            nn = rng.choice(NOUNS); q = rng.choice(Q_OTHER).format(noun=nn)
        elif case == "clarify_none":
            chat(rng.choice([1, 2]))
            nn = rng.choice(NOUNS); q = rng.choice(Q_PRON).format(noun=nn)
        msgs.append({"role": "user", "content": q})
        rows.append({"id": f"ref-{i:04d}", "case": case, "messages": msgs,
                     "target_project": target["name"] if target else None,
                     "noun": nn, "code": target["facts"].get(nn) if target else None})
    d = os.path.join(WORK, "reference_eval")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{out_name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pj = "projects_heldout.json" if heldout else "projects.json"
    json.dump({"projects": projects}, open(os.path.join(d, pj), "w"), indent=1)
    from collections import Counter
    print(len(rows), Counter(r["case"] for r in rows))


if __name__ == "__main__":
    main()
