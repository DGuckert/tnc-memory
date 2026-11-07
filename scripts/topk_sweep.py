"""Memory-note top_k sweep through the gateway model path.

Both structured shortcuts (fact_map lookup and structured_answer on the top note)
are disabled, so every query goes search -> rerank -> reader with k notes.

Two benchmarks:
  bench975  200 conversations over the 9.75M-token store (pronoun / explicit / typo).
            Every fact has 3 same-noun distractor notes in the store.
  hard      100 noisy multi-project conversations (real Discord chatter), store is the
            80 project notes only.

For bench975 the reader is run twice per k:
  ranked    notes in reranker order (what the gateway does)
  shuffled  same k notes, random order -> can the reader still find the gold note
            when it is not in slot 1?
"""
import argparse, json, os, random, re, statistics, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/daniel/tnc")
from gateway import MemoryGateway, extract_note

WORK = "/home/daniel/tnc/work"
N = 10485760
HARD_MIDDLEMAN = "/home/daniel/tnc/training_runs/middleman-lora"
SYSTEM = ("You are a memory-aware assistant. Resolve the user's project reference from the conversation. "
          "If a project is clear, call search_memory with the canonical project name and a self-contained "
          "query. If no project is clear, call ask_clarification. For small talk, call no_tool. "
          "Reply with JSON only.")


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if xs else None


def summ(xs):
    return {"median": pct(xs, 50), "p95": pct(xs, 95), "max": round(max(xs), 1) if xs else None,
            "mean": round(statistics.mean(xs), 1) if xs else None, "n": len(xs)}


def load_975():
    meta = json.load(open(os.path.join(WORK, f"needles_{N}.json"), encoding="utf-8"))
    facts = meta["facts"]
    docs = []
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    with open(os.path.join(WORK, f"mod_texts_{N}.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            docs[o["pos"]] = o["text"]
    cases = []
    for f in facts[:100]:
        cases.append(("pronoun", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"We're starting project {f['name']}."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"What's the {f['noun']} for it?"},
        ], f["code"]))
    for f in facts[100:150]:
        cases.append(("explicit", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"what is the {f['noun']} for project {f['name']}?"},
        ], f["code"]))
    for f in facts[150:200]:
        n = f["name"]
        typo = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        cases.append(("typo", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"We're starting project {n}."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"What's the {f['noun']} for {typo}?"},
        ], f["code"]))
    return docs, cases


def load_hard():
    meta = json.load(open(os.path.join(WORK, "hard_eval_meta.json"), encoding="utf-8"))
    code = {p["name"]: p["code"] for p in meta["projects"]}
    docs = [f"Note to self: the {p['noun']} for project {p['name']} is {p['code']}." for p in meta["projects"]]
    rows = [json.loads(l) for l in open(os.path.join(WORK, "hard_eval.jsonl"), encoding="utf-8")]
    cases = []
    for r in rows:
        t = r["target"]
        if t.get("tool") != "search_memory":
            continue  # non-search rows never reach the reader; k is irrelevant for them
        cases.append(("hard", r["messages"][:-1], code[t["args"]["project"]]))
    return docs, cases


def gold_rank(notes, gold):
    for i, n in enumerate(notes):
        if gold in n:
            return i + 1
    return None


def run(gw, docs, cases, ks, shuffle, seed=0):
    out = {}
    for k in ks:
        gw.top_k = k
        rng = random.Random(seed + k)
        for c in cases[:3]:
            gw.handle(c[1])
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        t_tot, t_read, t_rerank, toks = [], [], [], []
        ok = ok_shuf = present = ok_given_present = 0
        ranks = []
        by = {}
        for kind, msgs, gold in cases:
            res = gw.handle(msgs)
            notes = res.get("notes", [])
            r = gold_rank(notes, gold)
            hit = res.get("answer") == gold
            ok += hit
            if r:
                present += 1
                ok_given_present += hit
                ranks.append(r)
            b = by.setdefault(kind, {"n": 0, "ok": 0})
            b["n"] += 1; b["ok"] += hit
            tm = res.get("timings", {})
            if "reader" in tm:
                t_tot.append(tm["total"] * 1000); t_read.append(tm["reader"] * 1000)
                t_rerank.append(tm["rerank"] * 1000)
            if res.get("reader_prompt_tokens"):
                toks.append(res["reader_prompt_tokens"])
            row = {"kind": kind, "gold": gold, "answer": res.get("answer"), "gold_rank": r}
            if shuffle and res.get("ids"):
                ids = list(res["ids"]); rng.shuffle(ids)
                a2 = gw.answer(res.get("query", ""), ids)
                ok_shuf += a2 == gold
                row["answer_shuffled"] = a2
            rows.append(row)
        torch.cuda.synchronize()
        n = len(cases)
        out[str(k)] = {
            "k": k, "n": n,
            "accuracy": ok / n,
            "accuracy_shuffled": (ok_shuf / n) if shuffle else None,
            "gold_in_notes": present / n,
            "accuracy_given_gold_in_notes": ok_given_present / max(1, present),
            "gold_rank_median": statistics.median(ranks) if ranks else None,
            "gold_rank1_rate": sum(1 for x in ranks if x == 1) / n,
            "by_case": {kk: {**v, "acc": v["ok"] / v["n"]} for kk, v in by.items()},
            "reader_prompt_tokens": summ(toks),
            "latency_ms": {"total": summ(t_tot), "reader": summ(t_read), "rerank": summ(t_rerank)},
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "gpu_peak_over_weights_gb": round((torch.cuda.max_memory_allocated() - base) / 1e9, 3),
            "errors": [x for x in rows if x["answer"] != x["gold"]][:10],
        }
        o = out[str(k)]
        print(json.dumps({"k": k, "acc": o["accuracy"], "acc_shuf": o["accuracy_shuffled"],
                          "gold_in": o["gold_in_notes"], "tok_med": o["reader_prompt_tokens"]["median"],
                          "tok_max": o["reader_prompt_tokens"]["max"],
                          "lat_med": o["latency_ms"]["total"]["median"],
                          "lat_p95": o["latency_ms"]["total"]["p95"],
                          "gpu": o["gpu_peak_gb"]}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["bench975", "hard", "both"], default="both")
    ap.add_argument("--ks", default="1,5,10,20,50")
    ap.add_argument("--no-canon", action="store_true", help="reader gets the tool query, not the top-note rewrite")
    ap.add_argument("--out", default=os.path.join(WORK, "topk_sweep.json"))
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    canon = not a.no_canon
    result = {"ks": ks, "structured_shortcuts": False, "canonicalize": canon,
              "gpu": torch.cuda.get_device_name(0), "note": "reader LoRA was trained on 4-6 notes per prompt"}
    if a.bench in ("bench975", "both"):
        docs, cases = load_975()
        t0 = time.time()
        gw = MemoryGateway(docs, structured=False, search_k=50, canonicalize=canon)
        result["bench975"] = {"build_s": round(time.time() - t0, 1), "store_chunks": len(docs),
                              "sweep": run(gw, docs, cases, ks, shuffle=True)}
        del gw; torch.cuda.empty_cache()
        json.dump(result, open(a.out, "w"), indent=2)
    if a.bench in ("hard", "both"):
        docs, cases = load_hard()
        # same middleman for tool + reader as hard_gateway_eval.py; store has only 80 notes
        gw = MemoryGateway(docs, middleman=HARD_MIDDLEMAN, structured=False, search_k=50,
                           canonicalize=canon)
        result["hard"] = {"store_notes": len(docs), "sweep": run(gw, docs, cases, ks, shuffle=False)}
        json.dump(result, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
