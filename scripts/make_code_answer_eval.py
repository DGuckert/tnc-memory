import json, os, re, random
from collections import defaultdict

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work", "chat_coding")
SEED = 6161
CODE = re.compile(r"```|\b(def|class|import|pip|python|torch|cuda|gpu|error|traceback|bug|fix|code|function|api|json|sql|docker|linux|wsl|compile|install|model|token|prompt|context|quant|gguf|lora|finetune)\b", re.I)
QUESTION = re.compile(r"\?|\b(how|why|what|which|where|can|could|should|help|error|issue|problem|fix|install|run|use)\b", re.I)

def main():
    channels = defaultdict(list)
    for line in open(os.path.join(BASE, "corpus.jsonl"), encoding="utf-8"):
        o = json.loads(line)
        t = re.sub(r"\s+", " ", o["t"]).strip()
        if len(t) >= 8:
            channels[o["ch"]].append({"author": o["a"], "text": t, "ch": o["ch"]})
    rng = random.Random(SEED)
    for kind in ("code", "chat"):
        rows = []
        for ch, msgs in channels.items():
            for i, q in enumerate(msgs[:-1]):
                if not QUESTION.search(q["text"]):
                    continue
                target = None
                for j in range(i + 1, min(len(msgs), i + 5)):
                    t = msgs[j]["text"]
                    if len(t) < 30:
                        continue
                    is_code = bool(CODE.search(t)) or ch in ("local-llm-help", "local-llm-help-livechat")
                    if kind == "code" and is_code:
                        target = t; break
                    if kind == "chat" and not is_code:
                        target = t; break
                if target:
                    rows.append({"kind": kind, "query": q["text"], "target": target,
                                 "channel": ch, "target_author": msgs[j]["author"]})
        rng.shuffle(rows)
        print(kind, len(rows), flush=True)
        with open(os.path.join(WORK, f"answer_{kind}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows[:500]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
