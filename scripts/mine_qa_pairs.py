"""Mine instruction-style (question -> answer) pairs from the Discord export.

A pair is a help-seeking message (question mark or help phrasing, 40-600 chars) followed, within the
next 8 same-channel messages, by a substantive reply from a different author that either replies
to/mentions the asker or directly follows. Answers must be 60-1200 chars, mostly prose, and not
another question. Optional: stitch the asker's follow-up lines (same author, consecutive) onto the
question, since people split questions across messages.

Output stays under work/ (raw Discord text is never committed); only counts go in the repo.
"""
import json, re, collections

msgs = [json.loads(l) for l in open("/home/daniel/tnc/corpus.jsonl", encoding="utf-8")]
HELP = re.compile(r"\?|\b(how do i|how can i|anyone know|does anyone|is there a way|help|can't get|cannot get|"
                  r"keeps? (?:crashing|failing)|error|not working|recommend|should i|which (?:model|gpu|quant))\b", re.I)
CODEY = re.compile(r"\b(llama|cuda|gguf|vllm|python|pip|docker|quant|vram|kv|lora|api|server|config|flag|--|ctx|"
                   r"token|gpu|ram|model|ollama|lm studio|rocm|vulkan|install|error|exl|offload|moe)\b", re.I)

by_ch = collections.defaultdict(list)
for i, m in enumerate(msgs):
    by_ch[m["ch"]].append(i)

pairs = []
for ch, ids in by_ch.items():
    for p, i in enumerate(ids):
        q = msgs[i]
        qt = q["t"].strip()
        if not (40 <= len(qt) <= 600) or not HELP.search(qt) or "http" in qt:
            continue
        # stitch the asker's immediate continuation lines
        j = p + 1
        while j < len(ids) and msgs[ids[j]]["a"] == q["a"] and len(qt) < 900:
            qt += "\n" + msgs[ids[j]]["t"].strip()
            j += 1
        for k in range(j, min(len(ids), j + 8)):
            a = msgs[ids[k]]
            at = a["t"].strip()
            if a["a"] == q["a"]:
                continue
            if not (60 <= len(at) <= 1200) or at.endswith("?") or "http" in at[:20]:
                continue
            alpha = sum(c.isalpha() or c.isspace() for c in at) / len(at)
            if alpha < 0.75:
                continue
            pairs.append({"channel": ch, "q_idx": i, "a_idx": ids[k], "question": qt, "answer": at,
                          "technical": bool(CODEY.search(qt + " " + at)), "gap": k - j})
            break

seen = set()
uniq = []
for pr in pairs:
    key = pr["question"][:80].lower()
    if key not in seen:
        seen.add(key)
        uniq.append(pr)
tech = [p for p in uniq if p["technical"]]
with open("/home/daniel/tnc/work/qa_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in uniq:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
stats = {"pairs": len(uniq), "technical": len(tech),
         "by_channel": dict(collections.Counter(p["channel"] for p in uniq).most_common(8)),
         "technical_by_channel": dict(collections.Counter(p["channel"] for p in tech).most_common(8)),
         "median_q_chars": sorted(len(p["question"]) for p in uniq)[len(uniq) // 2],
         "median_a_chars": sorted(len(p["answer"]) for p in uniq)[len(uniq) // 2],
         "answer_directly_follows": round(sum(p["gap"] == 0 for p in uniq) / len(uniq), 3)}
json.dump(stats, open("/home/daniel/tnc-repo/results/training/qa_pairs_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
