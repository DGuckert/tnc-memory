import json, os, re, time, random, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

WORK = "/home/daniel/tnc/work"
OUT = os.path.join(WORK, "llm_results.json")
CHAT_OUT = os.path.join(WORK, "llm_chat_examples.json")
MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
DEVICE = "cuda"
SEED = 20261031
MAX_NEW = 24
SIZES = [(2000, 20), (4000, 40), (8000, 80)]

def log(*a):
    print(*a, flush=True)

def norm_code(s):
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()

def extract_codes(s):
    return re.findall(r"[A-Za-z0-9]{2,5}-[A-Za-z0-9]{3,5}", s)

def load_facts():
    data = json.load(open(os.path.join(WORK, "needles_131072.json"), encoding="utf-8"))
    return data["facts"]

def fact_lines(facts, n_needles, mode):
    recs = []
    for f in facts[:n_needles]:
        for label, name, code in [("gold", f["name"], f["code"])] + [
            ("distractor", d[0], d[1]) for d in f["distractors"]
        ]:
            if mode == "plain":
                line = f"Note to self: the {f['noun']} for project {name} is {code}."
            elif mode == "tags":
                line = f'<fact project="{name}" noun="{f["noun"]}" code="{code}"/>'
            elif mode == "index":
                line = f'project={name} | noun={f["noun"]} | code={code}'
            else:
                raise ValueError(mode)
            recs.append({"line": line, "needle_i": f["i"], "label": label,
                         "name": name, "noun": f["noun"], "code": code})
    return recs

def load_filler(tokenizer):
    lines = []
    with open(os.path.join(WORK, "chunks.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2500:
                break
            obj = json.loads(line)
            for ln in obj["text"].split("\n"):
                ln = ln.strip()
                if 3 <= len(ln) <= 300:
                    ids = tokenizer.encode(ln, add_special_tokens=False)
                    if 4 <= len(ids) <= 80:
                        lines.append(ln)
    rng = random.Random(SEED)
    rng.shuffle(lines)
    log("filler lines", len(lines))
    return lines

def ntok(tokenizer, s):
    return len(tokenizer.encode(s, add_special_tokens=False))

def build_history(tokenizer, facts, n_needles, mode, target_tokens, filler, max_ctx):
    recs = fact_lines(facts, n_needles, mode)
    fact_texts = [r["line"] for r in recs]
    fact_tok = sum(ntok(tokenizer, t) for t in fact_texts)
    overhead = 220
    budget = min(target_tokens, max_ctx - overhead)
    if fact_tok > budget:
        raise ValueError(f"facts too long: {fact_tok} > {budget} for n={n_needles} mode={mode}")
    selected = []
    total = fact_tok
    for ln in filler:
        n = ntok(tokenizer, ln)
        if total + n > budget:
            continue
        selected.append(ln)
        total += n
        if total >= budget - 60:
            break
    if mode == "index":
        history = "FACT INDEX:\n" + "\n".join(fact_texts) + "\n\nCONVERSATION:\n" + "\n".join(selected)
    else:
        blocks = fact_texts + selected
        random.Random(SEED + n_needles + (0 if mode == "plain" else 7)).shuffle(blocks)
        history = "\n".join(blocks)
    return history, fact_tok, total

def make_prompt(tokenizer, history, question):
    system = ("You are a helpful assistant with a long memory. Answer the user's question "
              "using the conversation history. If the answer is not in the history, say: I don't know.")
    user = history + "\n\nQuestion: " + question + "\nAnswer with the exact code."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    log("model", MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("chat template ok", bool(tokenizer.chat_template))
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model.to(DEVICE)
    model.eval()
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    log("loaded max_ctx", max_ctx, "mem_gb", round(torch.cuda.max_memory_allocated() / 1e9, 2))

    facts = load_facts()
    filler = load_filler(tokenizer)
    results = []
    chat_examples = []

    for target, n_needles in SIZES:
        for mode in ["plain", "tags", "index"]:
            log("===", target, mode, "needles", n_needles)
            history, fact_tok, hist_tok = build_history(
                tokenizer, facts, n_needles, mode, target, filler, max_ctx
            )
            log("history tokens", hist_tok, "fact tokens", fact_tok)
            qs = facts[:n_needles]
            rows = []
            for j, f in enumerate(qs):
                prompt = make_prompt(tokenizer, history, f["query"])
                enc = tokenizer(prompt, return_tensors="pt")
                ids = enc["input_ids"].to(DEVICE)
                attn = enc["attention_mask"].to(DEVICE)
                plen = int(ids.shape[1])
                if plen > max_ctx - 8:
                    log("SKIP too long", plen)
                    rows.append({"q": f["query"], "gold": f["code"], "correct": False,
                                 "skipped": True, "prompt_tokens": plen})
                    continue
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(
                        input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    )
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                ans = tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()
                gold_norm = norm_code(f["code"])
                ans_norm = norm_code(ans)
                correct = gold_norm in ans_norm
                codes = extract_codes(ans)
                wrong = (not correct) and any(norm_code(c) != gold_norm for c in codes)
                dk = ("don't know" in ans.lower()) or ("dont know" in ans.lower())
                rows.append({
                    "q": f["query"], "gold": f["code"], "answer": ans,
                    "correct": bool(correct), "wrong_code": bool(wrong),
                    "dont_know": bool(dk), "prompt_tokens": plen,
                    "gen_ms": dt * 1000.0,
                })
                if j < 8:
                    log(j, "correct", correct, "|", ans[:120].replace("\n", " "))
            acc = sum(r["correct"] for r in rows) / max(1, len(rows))
            dk_rate = sum(r.get("dont_know", False) for r in rows) / max(1, len(rows))
            wrong_rate = sum(r.get("wrong_code", False) for r in rows) / max(1, len(rows))
            ptoks = [r["prompt_tokens"] for r in rows if not r.get("skipped")]
            gms = sorted(r["gen_ms"] for r in rows if "gen_ms" in r)
            med = gms[len(gms) // 2] if gms else None
            results.append({
                "target_tokens": target, "mode": mode, "n_needles": n_needles,
                "history_tokens": hist_tok, "fact_tokens": fact_tok,
                "prompt_tokens_mean": sum(ptoks) / max(1, len(ptoks)),
                "prompt_tokens_max": max(ptoks) if ptoks else None,
                "accuracy": acc, "dont_know_rate": dk_rate, "wrong_code_rate": wrong_rate,
                "gen_ms_median": med, "rows": rows,
            })
            log("RESULT", target, mode, "acc", round(acc, 4), "dk", round(dk_rate, 4),
                "wrong", round(wrong_rate, 4), "median_ms", round(med, 1) if med else None)
            with open(OUT, "w", encoding="utf-8") as fh:
                json.dump({"model": MODEL, "results": results}, fh, indent=1)
            torch.cuda.empty_cache()

    prompts = [
        "Hello! How are you?",
        "Explain in one sentence what a transformer is.",
        "Write a haiku about the ocean.",
        "What is 17 * 23?",
        "Give me three tips for better sleep.",
        "Who are you?",
        "What is the capital of France?",
        "Tell me a short joke.",
    ]
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        chat_examples.append({"prompt": p, "answer": ans})
        log("CHAT", p, "=>", ans[:200].replace("\n", " "))
    with open(CHAT_OUT, "w", encoding="utf-8") as fh:
        json.dump({"model": MODEL, "examples": chat_examples}, fh, indent=1)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"model": MODEL, "results": results}, fh, indent=1)
    log("done")

if __name__ == "__main__":
    main()
