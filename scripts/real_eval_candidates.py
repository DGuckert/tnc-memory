"""Sample memorable Discord messages as targets for a hand-written recall eval.

Picks informative messages (80-400 chars, mostly words, not a link dump), stratified by channel,
and writes each with its 2 preceding same-channel messages for context. The questions themselves
are written separately (by Claude, before any retriever is tuned on them) -> silver labels.
"""
import json, random, re, collections

rng = random.Random(1107)
msgs = [json.loads(l) for l in open("/home/daniel/tnc/corpus.jsonl", encoding="utf-8")]


def good(t):
    if not (80 <= len(t) <= 400):
        return False
    if "http" in t or "```" in t:
        return False
    w = re.findall(r"[A-Za-z]{3,}", t)
    return len(w) >= 12 and sum(c.isalpha() or c == " " for c in t) / len(t) > 0.8


quota = {"general": 150, "llm-discussion": 60, "local-llm-help": 25, "local-llm-help-livechat": 25,
         "model-updates": 15, "off-topic": 10, "setup-showcase": 5}
by = collections.defaultdict(list)
for i, m in enumerate(msgs):
    if m["ch"] in quota and good(m["t"]):
        by[m["ch"]].append(i)
out = []
for ch, q in quota.items():
    for i in rng.sample(by[ch], min(q, len(by[ch]))):
        ctx = [j for j in range(i - 1, max(-1, i - 6), -1) if msgs[j]["ch"] == ch][:2][::-1]
        out.append({"idx": i, "ch": ch, "a": msgs[i]["a"], "t": msgs[i]["t"],
                    "ctx": [f"{msgs[j]['a']}: {msgs[j]['t'][:160]}" for j in ctx]})
rng.shuffle(out)
with open("/home/daniel/tnc/work/real_eval/candidates.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(len(out), {c: len(v) for c, v in by.items()})
