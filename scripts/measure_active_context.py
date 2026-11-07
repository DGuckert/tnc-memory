import json, os, re, random
import numpy as np
from transformers import AutoTokenizer

WORK = "/home/daniel/tnc/work"
TOK = "/home/daniel/tnc/training_runs/middleman-lora"
SYSTEM_TOOL = ("You are a memory-aware assistant. Resolve the user's project reference from the conversation. "
               "If a project is clear, call search_memory with the canonical project name and a self-contained "
               "query. If no project is clear, call ask_clarification. For small talk, call no_tool. "
               "Reply with JSON only.")
SYSTEM_READER = "You answer only from the memory notes. If the answer is not present, say exactly: I don't know."

def stats(xs):
    xs = np.array(xs)
    return {"n": len(xs), "min": int(xs.min()), "median": int(np.median(xs)),
            "p95": int(np.percentile(xs, 95)), "max": int(xs.max())}

def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    out = {}

    # 9.75M gateway tool prompts
    meta = json.load(open(os.path.join(WORK, "needles_10485760.json"), encoding="utf-8"))
    facts = meta["facts"]
    tool_lens = []
    for f in facts[:100]:
        msgs = [{"role": "system", "content": SYSTEM_TOOL},
                {"role": "user", "content": f"We're starting project {f['name']}."},
                {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": f"What's the {f['noun']} for it?"}]
        tool_lens.append(len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]))
    for f in facts[100:150]:
        msgs = [{"role": "system", "content": SYSTEM_TOOL},
                {"role": "user", "content": f"what is the {f['noun']} for project {f['name']}?"}]
        tool_lens.append(len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]))
    for f in facts[150:200]:
        n = f["name"]
        typo = (n[:-2] + n[-1] + n[-2]) if len(n) > 3 else (n + "x")
        msgs = [{"role": "system", "content": SYSTEM_TOOL},
                {"role": "user", "content": f"We're starting project {n}."},
                {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": f"What's the {f['noun']} for {typo}?"}]
        tool_lens.append(len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]))
    out["gateway_975m_tool_prompt_tokens"] = stats(tool_lens)

    # Reader prompt with 5 memory notes
    reader_lens = []
    rng = random.Random(7)
    for f in facts:
        others = rng.sample([g for g in facts if g is not f], 4)
        notes = [f"Note to self: the {g['noun']} for project {g['name']} is {g['code']}." for g in [f] + others]
        user = "Memory notes:\n" + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
        user += f"\nQuestion: what is the {f['noun']} for project {f['name']}?"
        msgs = [{"role": "system", "content": SYSTEM_READER}, {"role": "user", "content": user}]
        reader_lens.append(len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]))
    out["gateway_reader_prompt_tokens_5_notes"] = stats(reader_lens)

    # Hard noisy conversation tool prompts
    hard = [json.loads(line) for line in open(os.path.join(WORK, "hard_eval.jsonl"), encoding="utf-8")]
    hard_tool = []
    for r in hard:
        hard_tool.append(len(tok(tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]))
    out["hard_noisy_tool_prompt_tokens"] = stats(hard_tool)

    # Coding/chat adapter rows
    for name in ("chat_coding/train_chat_coding.jsonl", "chat_coding_clean/train_chat_coding.jsonl"):
        lens = []
        for i, line in enumerate(open(os.path.join(WORK, name), encoding="utf-8")):
            r = json.loads(line)
            prefix = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
            full = prefix + r["messages"][-1]["content"] + tok.eos_token
            lens.append(len(tok(full, add_special_tokens=False)["input_ids"]))
            if i >= 499:
                break
        out[f"chat_coding_adapter_tokens_{os.path.basename(os.path.dirname(name))}"] = stats(lens)

    json.dump(out, open(os.path.join(WORK, "active_context_sizes.json"), "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
