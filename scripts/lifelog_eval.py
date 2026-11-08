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
from lifelog_memory import FactStore, oracle_store, extract, store_accuracy, llm
import life_log as LL

host_reader_sweep.RAW = "1"
TODAY = "2026-10-31"
ANS_SYS = (f"You are the user's assistant. Today is {TODAY}. Answer the user's question about their projects using "
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
    {"name": "find_projects", "description": "List projects whose attribute equals a value, now or as of a date. "
     "attr is one of language, database, host, port, status, collaborator.", "parameters": {"type": "object",
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
            return ", ".join(r) if r else "none"
        if name == "list_projects":
            es = sorted(fs.entities(), key=lambda e: fs.value(e, "started") or "9999")
            return "\n".join(f"{e}: started {fs.value(e, 'started')}, status {fs.value(e, 'status')}" for e in es)
    except Exception as ex:
        return f"tool error: {ex}"
    return "unknown tool"


def agent(question, fs, searcher, max_steps=6):
    tools = "\n".join(json.dumps({"type": "function", "function": t}) for t in TOOLS)
    sys_ = (ANS_SYS.replace("using the memory below", "using the tools to look things up") +
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            f"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> "
            'XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')
    conv = [{"role": "system", "content": sys_}, {"role": "user", "content": question}]
    calls, ptoks, t0 = [], 0, time.perf_counter()
    for step in range(max_steps):
        text, pt, _ = llm(URL, conv, 400)
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
    ap.add_argument("--gate", action="store_true", help="keyword/entity gate before extraction (big log)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    URL = a.url
    msgs = [json.loads(l) for l in open(os.path.join(a.dir, "log.jsonl"), encoding="utf-8")]
    events = [tuple(e) for e in json.load(open(os.path.join(a.dir, "events.json")))]
    qs = json.load(open(os.path.join(a.dir, "questions.json")))
    if a.limit:
        qs = qs[:a.limit]
    if a.per_type:
        cnt = defaultdict(int)
        qs = [q for q in qs if (cnt.__setitem__(q["type"], cnt[q["type"]] + 1) or cnt[q["type"]] <= a.per_type)]
    conds = a.conds.split(",")
    out = {"dir": a.dir, "n_messages": len(msgs), "n_questions": len(qs), "conditions": {}}
    oracle = oracle_store(events)
    fs = None
    store_p = os.path.join(a.dir, "extracted_store.json")
    if any(c in ("cards", "agent") for c in conds):
        if os.path.exists(store_p):
            d = json.load(open(store_p))
            fs = FactStore()
            for e, attr, op, v, date in d["facts"]:
                fs.apply(e, attr, op, v, date)
            out["extraction"] = d["stats"]
        else:
            fs, stats = extract(msgs, URL, use_gate=a.gate, log=sys.stdout)
            facts = [(e, attr, "set", v, d) for (e, attr), h in fs.h.items() for d, v, _ in h if attr != "status" or True]
            facts += [(e, "collaborator", op, p, d) for e, h in fs.collab.items() for d, op, p, _ in h]
            facts.sort(key=lambda x: x[4])
            json.dump({"stats": stats, "facts": facts}, open(store_p, "w"))
            out["extraction"] = stats
        out["extraction"]["vs_oracle"] = store_accuracy(fs, oracle)
        print("extraction", json.dumps(out["extraction"]), flush=True)
    searcher = Searcher(msgs, a.dir) if any(c in ("rag", "cards", "cards_oracle", "agent") for c in conds) else None
    full_text = "\n".join(fmt_msg(m) for m in msgs)
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
            elif c in ("cards", "cards_oracle"):
                store = fs if c == "cards" else oracle
                ids = searcher.search(q["q"], 5)
                mem = ("Project cards (consolidated, current as of today):\n" + store.cards() +
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
