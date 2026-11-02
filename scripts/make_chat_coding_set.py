import argparse, json, os, random, re
from collections import defaultdict

BASE = "/home/daniel/tnc"
CORPUS = os.path.join(BASE, "corpus.jsonl")
SEED = 5150
CODING_HINTS = re.compile(r"```|\b(def|class|import|pip|python|torch|cuda|gpu|error|traceback|bug|fix|code|function|api|json|sql|docker|linux|wsl|compile|install|model|token|prompt|context|quant|gguf|lora|finetune)\b", re.I)
QUESTION_HINTS = re.compile(r"\?|\b(how|why|what|which|where|can|could|should|help|error|issue|problem|fix|install|run|use|code)\b", re.I)

def clean(s):
    s = s.replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BASE, "work", "chat_coding"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    channels = defaultdict(list)
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            t = clean(o["t"])
            if len(t) < 8 or len(t) > 1200:
                continue
            channels[o["ch"]].append({"author": o["a"], "text": t, "channel": o["ch"]})
    print("channels", len(channels), "messages", sum(len(v) for v in channels.values()), flush=True)

    rng = random.Random(SEED)
    eval_rows = []
    for ch, msgs in channels.items():
        for i in range(len(msgs) - 1):
            q = msgs[i]; t = msgs[i + 1]
            if not QUESTION_HINTS.search(q["text"]):
                continue
            kind = "coding" if (ch in ("local-llm-help", "local-llm-help-livechat") or CODING_HINTS.search(q["text"]) or CODING_HINTS.search(t["text"])) else "chat"
            eval_rows.append({"kind": kind, "query": q["text"], "target": t["text"],
                              "channel": ch, "query_author": q["author"], "target_author": t["author"]})
    for kind in ("coding", "chat"):
        pool = [r for r in eval_rows if r["kind"] == kind]
        rng.shuffle(pool)
        print(kind, len(pool), flush=True)
        with open(os.path.join(args.out, f"retrieval_{kind}.jsonl"), "w", encoding="utf-8") as f:
            for r in pool[:500]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sft_rows = []
    for ch, msgs in channels.items():
        for i in range(3, len(msgs)):
            target = msgs[i]
            context = msgs[max(0, i - 3):i]
            code = bool(CODING_HINTS.search(target["text"])) or ch in ("local-llm-help", "local-llm-help-livechat")
            kind = "coding" if code else "chat"
            sft_rows.append({"kind": kind, "channel": ch, "context": context, "target": target["text"]})
    for kind in ("coding", "chat"):
        pool = [r for r in sft_rows if r["kind"] == kind]
        rng.shuffle(pool)
        pool = pool[:6000]
        rows = []
        for i, r in enumerate(pool):
            ctx = "\n".join(f"{m['author']}: {m['text']}" for m in r["context"])
            system = ("You are a coding assistant in a Discord chat. Continue the conversation naturally and help with the code or technical question."
                      if kind == "coding" else
                      "You are participating in a Discord chat. Continue the conversation naturally.")
            rows.append({"id": f"{kind}-{i:06d}", "task": f"{kind}_completion",
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": f"Recent messages:\n{ctx}"},
                                      {"role": "assistant", "content": r["target"]}],
                         "target": r["target"], "meta": {"channel": r["channel"]}})
        channels_in = sorted({r["channel"] for r in pool})
        rng.shuffle(channels_in)
        n = len(channels_in)
        val_ch = set(channels_in[:max(1, int(n * 0.15))])
        test_ch = set(channels_in[max(1, int(n * 0.15)):max(2, int(n * 0.3))])
        train = [r for r in rows if r["meta"]["channel"] not in val_ch | test_ch]
        val = [r for r in rows if r["meta"]["channel"] in val_ch]
        test = [r for r in rows if r["meta"]["channel"] in test_ch]
        for name, part in (("train", train), ("val", val), ("test", test)):
            for r in part:
                r["id"] = f"{kind}-{name}-{r['id'].split('-')[-1]}"
                r["task"] = f"{kind}_completion"
            with open(os.path.join(args.out, f"{name}_{kind}.jsonl"), "w", encoding="utf-8") as f:
                for r in part:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(kind, "sft", len(train), len(val), len(test), "channels", n, flush=True)
    print("wrote", args.out, flush=True)

if __name__ == "__main__":
    main()
