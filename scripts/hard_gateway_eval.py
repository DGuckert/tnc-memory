import json, os, sys
sys.path.insert(0, "/home/daniel/tnc")
from gateway import MemoryGateway, extract_note

WORK = "/home/daniel/tnc/work"
MIDDLEMAN = "/home/daniel/tnc/training_runs/middleman-lora"
RERANKER = "/home/daniel/tnc/training_runs/reranker-mini"

def main():
    meta = json.load(open(os.path.join(WORK, "hard_eval_meta.json"), encoding="utf-8"))
    projects = meta["projects"]
    code = {p["name"]: p["code"] for p in projects}
    docs = [f"Note to self: the {p['noun']} for project {p['name']} is {p['code']}." for p in projects]
    rows = [json.loads(line) for line in open(os.path.join(WORK, "hard_eval.jsonl"), encoding="utf-8")]
    gw = MemoryGateway(docs, middleman=MIDDLEMAN, reranker=RERANKER)

    tool_ok = 0; e2e_ok = 0; search_n = 0; search_ok = 0
    other_n = 0; other_ok = 0
    examples = []
    for r in rows:
        target = r["target"]
        res = gw.handle(r["messages"][:-1])
        pred = res.get("tool")
        t_tool = target.get("tool")
        ok_tool = isinstance(pred, dict) and pred.get("tool") == t_tool
        if ok_tool and t_tool == "search_memory":
            pa = pred.get("args", {}); ta = target.get("args", {})
            ok_tool = pa.get("project") == ta.get("project") and pa.get("query") == ta.get("query")
        if ok_tool:
            tool_ok += 1
        if t_tool == "search_memory":
            search_n += 1
            want = code[target["args"]["project"]]
            ans = res.get("answer")
            if ans == want:
                search_ok += 1
                e2e_ok += 1
        else:
            other_n += 1
            if ok_tool:
                other_ok += 1
                e2e_ok += 1
        if len(examples) < 8:
            examples.append({"target": target, "pred": pred, "answer": res.get("answer"),
                             "mode": res.get("mode"), "query": res.get("query")})
    n = len(rows)
    out = {
        "n": n, "tool_full_acc": tool_ok / n, "end_to_end_acc": e2e_ok / n,
        "search_n": search_n, "search_answer_acc": search_ok / max(1, search_n),
        "other_n": other_n, "other_action_acc": other_ok / max(1, other_n),
        "examples": examples,
    }
    json.dump(out, open(os.path.join(WORK, "hard_gateway_metrics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
