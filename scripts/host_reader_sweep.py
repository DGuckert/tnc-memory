"""top_k sweep with a real host model reading the middleman's notes (memory_api.retrieve).

The structured shortcut is ignored here on purpose: the host must find the value in the notes.
Reference resolution comes from EntityTracker (not the old regex), retrieval from the gateway's
search + cross-encoder rerank, then k notes -> host via an OpenAI-compatible endpoint.

  ranked    notes in reranker order
  shuffled  same notes, random order (is the host actually reading, or copying note 1?)
"""
import argparse, json, os, random, statistics, sys, time, urllib.request
import numpy as np
sys.path.insert(0, "/home/daniel/tnc")
from memory_api import Memory
from topk_sweep import load_975, load_hard

SYSTEM = ("You are an assistant with access to memory notes retrieved for this conversation. "
          "Answer the user's last question using the notes. Reply with only the requested value. "
          "If the value is not in the notes, reply exactly: I don't know.")


# Qwen3 "Thinking" checkpoints ignore enable_thinking=False (their template always opens <think>).
# HOST_RAW_CHATML=1 builds the ChatML prompt by hand with an empty, already-closed think block and
# sends it to /completion, so the model answers directly. HOST_RAW_CHATML=think leaves thinking on.
RAW = os.environ.get("HOST_RAW_CHATML", "")


def chatml(msgs, think=False):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return s + "<|im_start|>assistant\n" + ("<think>\n" if think else "<think>\n\n</think>\n\n")


def complete(url, msgs, max_tokens, cache_prompt=True):
    """Returns (text, prompt_tokens, timings, wall_ms)."""
    t = time.perf_counter()
    if RAW:
        think = RAW == "think"
        body = {"prompt": chatml(msgs, think), "n_predict": max_tokens * (40 if think else 1), "temperature": 0,
                "cache_prompt": cache_prompt, "stop": ["<|im_end|>"]}
        o = json.loads(urllib.request.urlopen(urllib.request.Request(
            url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
        text = o["content"]
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        tm = o.get("timings", {})
        # prompt_n excludes the cached prefix; tokens_evaluated is the whole prompt
        return text.strip(), o.get("tokens_evaluated") or tm.get("prompt_n"), tm, (time.perf_counter() - t) * 1e3
    body = {"model": "host", "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": cache_prompt, "chat_template_kwargs": {"enable_thinking": False}}
    o = json.loads(urllib.request.urlopen(urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read())
    return (o["choices"][0]["message"]["content"].strip(), o.get("usage", {}).get("prompt_tokens"),
            o.get("timings", {}), (time.perf_counter() - t) * 1e3)


def ask(url, model, msgs, max_tokens=24):
    if RAW:
        text, pt, _, ms = complete(url, msgs, max_tokens)
        return text, pt, ms
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.perf_counter()
    o = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return o["choices"][0]["message"]["content"].strip(), o.get("usage", {}).get("prompt_tokens"), (time.perf_counter() - t) * 1e3


def build_msgs(conv, ret, notes):
    turns = [m for m in conv if m["role"] != "system"][-6:]
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in turns[:-1])
    hint = f"(The user is referring to project {ret['entity']}.)\n" if ret["entity"] else ""
    user = (f"Memory notes:\n{Memory.format_context(notes)}\n\nRecent conversation:\n{convo}\n\n"
            f"{hint}Question: {turns[-1]['content']}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8131")
    ap.add_argument("--model", default="host")
    ap.add_argument("--tag", default="gemma3-1b")
    ap.add_argument("--ks", default="1,5,10,20,50")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bench", default="bench975,hard")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    out = {"host": a.tag, "ks": ks}
    for bench in a.bench.split(","):
        docs, cases = load_975() if bench == "bench975" else load_hard()
        if a.limit:
            cases = cases[:a.limit]
        mem = Memory(fact_docs=docs)
        res = {}
        # retrieve once at max k; smaller k = prefix of the same ranking
        rets = []
        t_ret = []
        for kind, conv, gold in cases:
            msgs = [m for m in conv if m["role"] != "system"]
            t = time.perf_counter()
            r = mem.retrieve(msgs, k=max(ks), sources=("fact",))
            t_ret.append((time.perf_counter() - t) * 1e3)
            rets.append(r)
        for k in ks:
            rng = random.Random(k)
            ok = ok_s = gold_in = clar = 0
            toks, lat = [], []
            errs = []
            for (kind, conv, gold), r in zip(cases, rets):
                if r["needs_clarification"]:
                    clar += 1
                    continue
                notes = r["notes"][:k]
                gold_in += any(gold in n["text"] for n in notes)
                ans, pt, ms = ask(a.url, a.model, build_msgs(conv, r, notes))
                hit = gold in ans
                ok += hit
                toks.append(pt or 0); lat.append(ms)
                sh = notes[:]; rng.shuffle(sh)
                ans2, _, _ = ask(a.url, a.model, build_msgs(conv, r, sh))
                ok_s += gold in ans2
                if not hit and len(errs) < 8:
                    errs.append({"kind": kind, "gold": gold, "answer": ans, "entity": r["entity"]})
            n = len(cases)
            res[str(k)] = {"k": k, "n": n, "accuracy": ok / n, "accuracy_shuffled": ok_s / n,
                           "gold_in_notes": gold_in / n, "clarified": clar,
                           "prompt_tokens": {"median": pct(toks, 50), "p95": pct(toks, 95), "max": max(toks) if toks else None},
                           "host_latency_ms": {"median": pct(lat, 50), "p95": pct(lat, 95)},
                           "errors": errs}
            print(bench, json.dumps({kk: v for kk, v in res[str(k)].items() if kk != "errors"}), flush=True)
        out[bench] = {"retrieve_ms": {"median": pct(t_ret, 50), "p95": pct(t_ret, 95)}, "sweep": res}
        del mem
    p = f"/home/daniel/tnc/work/host_reader_{a.tag}.json"
    json.dump(out, open(p, "w"), indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
