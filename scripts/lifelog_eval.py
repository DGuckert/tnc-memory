"""Does the memory layer behave like context? Life-log questions under five conditions.

  full          the whole log in the prompt (true context; only possible for the small log)
  rag           top-10 messages from hybrid search + rerank (what v2 retrieve() does for messages)
  cards         write-time consolidated state: every project card (+ top-5 messages)
  cards_oracle  same, but cards built from the gold events (upper bound: perfect extraction)
  agent         host calls tools in a loop: search_messages, get_project, find_projects, list_projects

  python lifelog_eval.py --dir work/lifelog/small --conds full,rag,cards,cards_oracle,agent
"""
import argparse, json, os, re, sys, time
from collections import defaultdict
sys.path.insert(0, "/home/daniel/tnc")
import host_reader_sweep
from host_reader_sweep import complete
from lifelog_memory import FactStore, oracle_store, extract, store_accuracy, llm, consolidate, recall_pass
import life_log as LL

host_reader_sweep.RAW = "1"
TODAY = "2026-10-31"
ANS_SYS = (f"You are the user's assistant. Today is {TODAY}. Answer the user's question about their life and projects using "
           "the memory below; later information overrides earlier information. Think briefly if you need to, then "
           "finish with one final line 'ANSWER: <answer>' where the answer is a single value, a comma-separated list "
           "of names, or a number. If the memory does not contain it, write 'ANSWER: I don't know'.")


def final_answer(url, msgs, max_tokens=400):
    """Qwen3-Thinking reasons out loud even with thinking off, so let it, then take the last ANSWER line.
    If it runs out of tokens first, continue the same generation from '\\nANSWER:' for a short tail."""
    text, pt, ms = llm(url, msgs, max_tokens)
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if m:
        return m[-1].strip(), text, pt, ms
    tail, _, ms2 = llm(url, msgs, 40, prefix=text + "\nANSWER:")
    return tail.strip().split("\n")[0], text + "\nANSWER:" + tail, pt, ms + ms2


def fmt_msg(m):
    return f"[{m['date']}] {m['a']}: {m['t']}"


# ---------------- scoring ----------------
WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                      "thirteen fourteen fifteen sixteen".split())}
DOMAINS = [LL.PROJECTS, LL.PEOPLE, LL.HOSTS, LL.LANGS, LL.DBS, list(LL.STATUSES)]


def find_all(ans, items):
    out = []
    for it in items:
        pat = r"(?<![\w+])" + re.escape(it) + r"(?![\w+])"
        for m in re.finditer(pat, ans, re.I):
            out.append((m.start(), it))
    return sorted(out)


def domain_of(v):
    v = str(v)
    if v.isdigit():
        return None
    for d in DOMAINS:
        if v.lower() in (x.lower() for x in d):
            return d
    return [v]


def score(q, ans):
    g, kind = q["gold"], q["kind"]
    a = ans or ""
    if kind == "count":
        m = re.search(r"\b(\d+)\b", a)
        n = int(m.group(1)) if m else next((v for w, v in WORDNUM.items() if re.search(rf"\b{w}\b", a, re.I)), None)
        return float(n == g)
    if kind == "value":
        if str(g).isdigit():
            nums = re.findall(r"\b\d{4}\b", a)
            return float(bool(nums) and nums[0] == str(g))
        hits = find_all(a, domain_of(g))
        return float(bool(hits) and hits[0][1].lower() == str(g).lower())
    if kind == "set":
        hits = {h[1].lower() for h in find_all(a, domain_of(g[0]))}
        return float(hits == {x.lower() for x in g})
    if kind == "keywords":
        # every group must match; any alternative in a group counts. Words match on a prefix boundary
        # ('exam' matches 'exams'), numbers on full boundaries ('29' does not match '290').
        def hit(alt):
            pat = (r"(?<!\d)" + re.escape(alt) + r"(?!\d)") if alt.isdigit() else (r"(?<![A-Za-z])" + re.escape(alt))
            return re.search(pat, a, re.I) is not None
        if re.search(r"\bi don'?t know\b", a, re.I):
            return 0.0
        return float(all(any(hit(x) for x in grp) for grp in g))
    if kind == "sequence":
        if g[0].isdigit():
            seq = re.findall(r"\b\d{4}\b", a)
        else:
            seq = [h[1].lower() for h in find_all(a, domain_of(g[0]))]
        dedup = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]
        return float(dedup == [x.lower() for x in g])
    return 0.0


# ---------------- conditions ----------------
def ask(memory_text, question):
    msgs = [{"role": "system", "content": ANS_SYS},
            {"role": "user", "content": f"Memory:\n{memory_text}\n\nQuestion: {question}"}]
    ans, raw, pt, ms = final_answer(URL, msgs)
    return ans, pt, raw, ms


class Searcher:
    def __init__(self, msgs, work):
        import message_index
        message_index.WORK = work  # embedding cache lives next to this log, never mixed with the corpus cache
        self.mi = message_index.MessageIndex(msgs=msgs, device="cuda", rerank=True)
        self.msgs = msgs

    def search(self, q, k=10):
        ids = [i for i, _ in self.mi.search(q, k=k, mode="hybrid", rerank=True)]
        return sorted(ids)


TOOLS = [
    {"name": "search_messages", "description": "Full-text + semantic search over every chat message the user has "
     "ever sent. Returns dated messages.", "parameters": {"type": "object", "properties": {
         "query": {"type": "string"}, "k": {"type": "integer", "default": 8}}, "required": ["query"]}},
    {"name": "get_project", "description": "Everything remembered about one project: current language, database, "
     "host, port, status and collaborators, with the full history of changes and dates.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "find_projects", "description": "Count and list the projects whose attribute equals a value, now or as of "
     "a date. Exact: use it for any 'which' or 'how many' question. attr is one of language, database, host, port, "
     "status, collaborator.", "parameters": {"type": "object",
     "properties": {"attr": {"type": "string"}, "value": {"type": "string"},
                    "as_of": {"type": "string", "description": "YYYY-MM-DD, optional"}}, "required": ["attr", "value"]}},
    {"name": "list_projects", "description": "Every project with its start date and current status, oldest first.",
     "parameters": {"type": "object", "properties": {}}},
]


def run_tool(name, args, fs, searcher):
    try:
        if name == "search_messages":
            ids = searcher.search(args["query"], k=int(args.get("k", 8)))
            return "\n".join(fmt_msg(searcher.msgs[i]) for i in ids) or "no results"
        if name == "get_project":
            n = args["name"].strip()
            hit = next((e for e in fs.entities() if e.lower() == n.lower()), None)
            return fs.card(hit) if hit else f"no project named {n}; known: {', '.join(fs.entities())}"
        if name == "find_projects":
            r = fs.query(args["attr"].lower(), args["value"], args.get("as_of") or None)
            return f"{len(r)} project(s): " + ", ".join(r) if r else "0 projects"
        if name == "list_projects":
            es = sorted(fs.entities(), key=lambda e: fs.value(e, "started") or "9999")
            return "\n".join(f"{e}: started {fs.value(e, 'started')}, status {fs.value(e, 'status')}" for e in es)
    except Exception as ex:
        return f"tool error: {ex}"
    return "unknown tool"


AGENT_GRAMMAR = r'''
root  ::= note act
note  ::= [^<]{0,500}
act   ::= call | ans
call  ::= "<tool_call>\n{\"name\": \"" fname "\", \"arguments\": {" args? "}}\n</tool_call>"
fname ::= "search_messages" | "get_project" | "find_projects" | "list_projects"
args  ::= kv (", " kv){0,3}
kv    ::= "\"" [a-z_]{1,10} "\": " ( "\"" [^"\n]{0,80} "\"" | [0-9]{1,4} )
ans   ::= "ANSWER: " [^\n<]{1,200}
'''


def agent(question, fs, searcher, max_steps=8):
    tools = "\n".join(json.dumps({"type": "function", "function": t}) for t in TOOLS)
    sys_ = (ANS_SYS.replace("using the memory below", "using the tools to look things up") +
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            f"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> "
            'XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')
    conv = [{"role": "system", "content": sys_}, {"role": "user", "content": question}]
    calls, ptoks, t0 = [], 0, time.perf_counter()
    for step in range(max_steps):
        # The Thinking checkpoint deliberates for 400+ tokens about which tool to use and never calls one.
        # The grammar allows a short note (<=500 chars), then forces an action: a tool call or the answer.
        # Tried and rejected (2026-11-08, 26 previously-failed questions): no note at all -> 0/12, the model
        # answers "I don't know" without calling anything; forcing step 0 to be a tool call -> 1/26, it picks
        # the wrong tool (find_projects(name, X)) and gives up. This checkpoint needs room to think to use tools.
        text, pt, _ = llm(URL, conv, 320, grammar=AGENT_GRAMMAR)
        ptoks += pt or 0
        tc = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
        if not tc:
            m = re.findall(r"ANSWER:\s*(.+)", text)
            if m:
                return m[-1].strip(), calls, ptoks, (time.perf_counter() - t0) * 1e3
            tail, _, _ = llm(URL, conv, 40, prefix=text + "\nANSWER:")
            return tail.strip().split("\n")[0], calls, ptoks, (time.perf_counter() - t0) * 1e3
        conv.append({"role": "assistant", "content": text})
        resp = []
        for raw in tc:
            try:
                o = json.loads(raw)
                name, args = o["name"], o.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
            except Exception:
                resp.append("<tool_response>\ncould not parse tool call\n</tool_response>")
                continue
            out = run_tool(name, args, fs, searcher)
            calls.append({"name": name, "args": args})
            resp.append(f"<tool_response>\n{out[:4000]}\n</tool_response>")
        conv.append({"role": "user", "content": "\n".join(resp)})
    conv.append({"role": "user", "content": "Answer now with what you have, ending with 'ANSWER: <answer>'."})
    ans, _, pt, _ = final_answer(URL, conv, 200)
    return ans, calls, ptoks + (pt or 0), (time.perf_counter() - t0) * 1e3


def main():
    global URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--url", default="http://172.24.240.1:8131")
    ap.add_argument("--conds", default="full,rag,cards,cards_oracle,agent")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per_type", type=int, default=0)
    ap.add_argument("--recall", action="store_true", help="second read of project messages that left no facts")
    ap.add_argument("--gate", action="store_true", help="keyword/entity gate before extraction (big log)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--open", action="store_true", help="extract open-ended facts too (life_log_open logs)")
    ap.add_argument("--types", default="", help="comma list of question types to run (default: all)")
    a = ap.parse_args()
    URL = a.url
    msgs = [json.loads(l) for l in open(os.path.join(a.dir, "log.jsonl"), encoding="utf-8")]
    events = [tuple(e) for e in json.load(open(os.path.join(a.dir, "events.json")))]
    qs = json.load(open(os.path.join(a.dir, "questions.json")))
    if a.types:
        qs = [q for q in qs if q["type"] in a.types.split(",")]
    if a.limit:
        qs = qs[:a.limit]
    if a.per_type:
        cnt = defaultdict(int)
        qs = [q for q in qs if (cnt.__setitem__(q["type"], cnt[q["type"]] + 1) or cnt[q["type"]] <= a.per_type)]
    conds = a.conds.split(",")
    out = {"dir": a.dir, "n_messages": len(msgs), "n_questions": len(qs), "conditions": {}}
    oracle = oracle_store(events)
    fs = fs_raw = None
    raw_p = os.path.join(a.dir, "extracted_raw.json")
    if any(c in ("cards", "cards_v2", "agent", "mem") for c in conds):
        if os.path.exists(raw_p):
            d = json.load(open(raw_p))
            stats, raw = d["stats"], d["raw"]
        else:
            _, stats = extract(msgs, URL, use_gate=a.gate, log=sys.stdout)
            raw = stats.pop("raw")
            json.dump({"stats": stats, "raw": raw}, open(raw_p, "w"))
        fs_raw = consolidate(raw, msgs, hygiene=False)
        if a.recall:
            # second, closer read of messages that name a project but left nothing in the store
            d = json.load(open(raw_p))
            if "recall" not in d:
                extra, rst = recall_pass(msgs, d["raw"], consolidate(d["raw"], msgs), URL, use_gate=a.gate,
                                         log=sys.stdout)
                d["recall"] = {"stats": rst, "raw": extra}
                json.dump(d, open(raw_p, "w"))
            raw = raw + d["recall"]["raw"]
            stats = dict(stats, **d["recall"]["stats"])
        if a.open:
            # separate open-facts pass (a combined prompt pushed everything into project slots)
            d = json.load(open(raw_p))
            if "open" not in d:
                from lifelog_memory import extract_open, open_targets
                oraw, ost_ = extract_open(msgs, URL, idx=open_targets(msgs) if a.gate else None, log=sys.stdout)
                d["open"] = {"stats": ost_, "raw": oraw}
                json.dump(d, open(raw_p, "w"))
            raw = raw + d["open"]["raw"]
            stats = dict(stats, **d["open"]["stats"])
        hyg = {}
        fs = consolidate(raw, msgs, hygiene=True, stats=hyg)
        hyg.pop("reattach_log", None)
        out["extraction"] = dict(stats, hygiene=hyg, vs_oracle_raw=store_accuracy(fs_raw, oracle),
                                 vs_oracle=store_accuracy(fs, oracle))
        print("extraction", json.dumps(out["extraction"]), flush=True)
    searcher = Searcher(msgs, a.dir) if any(c != "full" for c in conds) else None
    full_text = "\n".join(fmt_msg(m) for m in msgs)
    open_store = open_oracle = None
    if any(c in ("mem", "mem_oracle") for c in conds):
        from open_memory import consolidate_open, oracle_open
        emb_model = searcher.mi.emb
        embed = lambda s: emb_model.encode([s], normalize_embeddings=True)[0]
        open_oracle = oracle_open(json.load(open(os.path.join(a.dir, "open_events.json"))), msgs, embed)
        if "mem" in conds:
            ost = {}
            open_store = consolidate_open(raw, msgs, embed=embed, stats=ost, typed_entities=fs.entities())
            # something held as a value of a state elsewhere ('me | keyboard | Corne') is not a project
            fs.exclude = {f["val"] for f in open_store.facts if f["kind"] == "state"}
            out["extraction"]["vs_oracle_with_open_exclude"] = store_accuracy(fs, oracle)
            out.setdefault("extraction", {})["open_hygiene"] = ost
            out["extraction"]["open_subjects"] = len(open_store.subjects())
            print("open store", json.dumps(ost), "subjects:", open_store.subjects()[:40], flush=True)
        fact_search = {}
        for name, st_ in (("mem", open_store), ("mem_oracle", open_oracle)):
            if st_ is None:
                continue
            import message_index, glob
            fdir = os.path.join(a.dir, f"factidx_{name}{a.tag}")
            os.makedirs(fdir, exist_ok=True)
            for f in glob.glob(os.path.join(fdir, "emb_*.npy")):  # facts change between runs: never reuse
                os.remove(f)
            sents = st_.sentences()
            fmsgs = [{"ch": "facts", "a": st_.facts[i]["subj"], "t": t} for i, t in sents]
            message_index.WORK = fdir
            fact_search[name] = (message_index.MessageIndex(msgs=fmsgs, device="cuda", rerank=True), sents)
            message_index.WORK = a.dir

    def open_memory_text(store, name, question):
        from open_memory import mentioned_subjects
        subs = mentioned_subjects(question, store)
        mi, sents = fact_search[name]
        hits = [sents[i][1] for i, _ in mi.search(question, k=12, mode="hybrid", rerank=True)]
        cards = "\n".join(store.card(s) for s in subs)
        return (("\n\nPeople, preferences and other facts (cards for what the question mentions):\n" + cards
                 if cards else "") + "\n\nRelated remembered facts:\n" + "\n".join(hits))
    for c in conds:
        rows = []
        for q in qs:
            t0 = time.perf_counter()
            calls = None
            if c == "full":
                ans, pt, _, _ = ask(full_text, q["q"])
            elif c == "rag":
                ids = searcher.search(q["q"], 10)
                ans, pt, _, _ = ask("\n".join(fmt_msg(msgs[i]) for i in ids), q["q"])
            elif c in ("cards", "cards_oracle", "cards_v2", "cards_v2_oracle"):
                # cards: raw extraction, cards only. cards_v2: hygiene rules + write-time aggregate views.
                store = {"cards": fs_raw, "cards_oracle": oracle, "cards_v2": fs, "cards_v2_oracle": oracle}[c]
                ids = searcher.search(q["q"], 5)
                views = ("\n\nIndexes (precomputed from the cards, current as of today):\n" + store.views()
                         if "v2" in c else "")
                mem = ("Project cards (consolidated, current as of today):\n" + store.cards() + views +
                       "\n\nRelevant messages:\n" + "\n".join(fmt_msg(msgs[i]) for i in ids))
                ans, pt, _, _ = ask(mem, q["q"])
            elif c in ("mem", "mem_oracle"):
                # v4: typed project cards + indexes (v3) + open-fact cards for mentioned subjects + fact search
                store = fs if c == "mem" else oracle
                ostore = open_store if c == "mem" else open_oracle
                ids = searcher.search(q["q"], 5)
                mem = ("Project cards (consolidated, current as of today):\n" + store.cards() +
                       "\n\nIndexes (precomputed from the cards, current as of today):\n" + store.views() +
                       open_memory_text(ostore, c, q["q"]) +
                       "\n\nRelevant messages:\n" + "\n".join(fmt_msg(msgs[i]) for i in ids))
                ans, pt, _, _ = ask(mem, q["q"])
            elif c == "agent":
                ans, calls, pt, _ = agent(q["q"], fs, searcher)
            ms = (time.perf_counter() - t0) * 1e3
            s = score(q, ans)
            rows.append({"id": q["id"], "type": q["type"], "score": s, "answer": ans[:300], "gold": q["gold"],
                         "prompt_tokens": pt, "ms": round(ms), "calls": calls})
        by = defaultdict(list)
        for r in rows:
            by[r["type"]].append(r["score"])
        summ = {"accuracy": round(sum(r["score"] for r in rows) / len(rows), 3),
                "by_type": {t: round(sum(v) / len(v), 3) for t, v in by.items()},
                "prompt_tokens_median": sorted(r["prompt_tokens"] or 0 for r in rows)[len(rows) // 2],
                "latency_ms_median": sorted(r["ms"] for r in rows)[len(rows) // 2]}
        out["conditions"][c] = {"summary": summ, "rows": rows}
        print(c, json.dumps(summ), flush=True)
        p = os.path.join("/home/daniel/tnc/work/lifelog", f"lifelog_eval_{os.path.basename(a.dir.rstrip('/'))}{a.tag}.json")
        json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
