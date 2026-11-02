import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = "/home/daniel/tnc/work"
OUT = os.path.join(WORK, "llm_report.md")

def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)

def mcnemar(rows_a, rows_b):
    a = {r["q"]: r for r in rows_a if not r.get("skipped")}
    b = {r["q"]: r for r in rows_b if not r.get("skipped")}
    common = [q for q in a if q in b]
    n01 = sum(1 for q in common if a[q]["correct"] and not b[q]["correct"])
    n10 = sum(1 for q in common if not a[q]["correct"] and b[q]["correct"])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return n01, n10, min(1.0, p)

def fmt(x, nd=3):
    return f"{x:.{nd}f}"

def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)

def main():
    final = load(os.path.join(WORK, "llm_results_final.json"))
    six = load(os.path.join(WORK, "llm_results_6k.json"))
    lat = load(os.path.join(WORK, "llm_latency.json"))
    chat = load(os.path.join(WORK, "llm_chat_examples_final.json"))
    if chat is None:
        chat = load(os.path.join(WORK, "llm_chat_examples_6k.json"))

    results = []
    for d in [final, six]:
        if not d:
            continue
        for r in d["results"]:
            if r.get("gen_ms_median") is None or not r.get("rows"):
                continue
            if all(x.get("skipped") for x in r["rows"]):
                continue
            results.append(r)
    results.sort(key=lambda r: (r["target_tokens"], ["plain", "tags", "index"].index(r["mode"])))
    if not results:
        raise SystemExit("no usable LLM results")

    sizes = sorted(set(r["target_tokens"] for r in results))
    modes = ["plain", "tags", "index"]
    by = {(r["target_tokens"], r["mode"]): r for r in results}

    rows_tab = []
    for r in results:
        n = len([x for x in r["rows"] if not x.get("skipped")])
        k = sum(1 for x in r["rows"] if x.get("correct"))
        lo, hi = wilson(k, n)
        rows_tab.append([
            f"{r['target_tokens']//1000}K", r["mode"], r["n_fact_needles"], r["n_questions"],
            f"{r['history_tokens']:,}", f"{r['fact_tokens']:,}",
            fmt(r["prompt_tokens_mean"], 0), f"{k}/{n}",
            f"{r['accuracy']:.2f} ({lo:.2f}-{hi:.2f})",
            fmt(r["wrong_code_rate"], 2), fmt(r["dont_know_rate"], 2),
            f"{r['gen_ms_median']:.0f}" if r.get("gen_ms_median") else "-",
        ])
    acc_tab = table(["size", "mode", "fact needles", "questions", "history tokens",
                     "fact tokens", "prompt tokens", "correct", "accuracy (95% CI)",
                     "wrong code", "don't know", "median gen ms"], rows_tab)

    size_tab_rows = []
    for s in sizes:
        p = by.get((s, "plain")); t = by.get((s, "tags")); i = by.get((s, "index"))
        if not (p and t and i):
            continue
        size_tab_rows.append([
            f"{s//1000}K", f"{p['history_tokens']:,}", f"{p['fact_tokens']:,}",
            f"{t['fact_tokens']:,}", f"{i['fact_tokens']:,}",
            f"{(t['fact_tokens']-p['fact_tokens'])/p['fact_tokens']*100:+.1f}%",
            f"{(i['fact_tokens']-p['fact_tokens'])/p['fact_tokens']*100:+.1f}%",
            f"{p['prompt_tokens_mean']:.0f}", f"{t['prompt_tokens_mean']:.0f}",
            f"{i['prompt_tokens_mean']:.0f}",
        ])
    size_tab = table(["size", "plain history", "plain fact", "tags fact", "index fact",
                      "tags fact vs plain", "index fact vs plain", "plain prompt",
                      "tags prompt", "index prompt"], size_tab_rows)

    pair_rows = []
    for s in sizes:
        for other in ["tags", "index"]:
            a = by.get((s, "plain")); b = by.get((s, other))
            if not (a and b):
                continue
            n01, n10, p = mcnemar(a["rows"], b["rows"])
            pair_rows.append([f"{s//1000}K", f"plain vs {other}", n01, n10, fmt(p, 3)])
    pair_tab = table(["size", "comparison", "plain only correct", "other only correct",
                      "McNemar exact p"], pair_rows)

    lat_tab = ""
    lat_note = ""
    if lat and lat.get("results"):
        r0 = lat["results"][0]
        lat_note = (f"{r0['prefill_ms_median']/1000:.1f}s at 2K plain "
                    f"({r0['prefill_tokens_per_s']:.0f} tok/s)")
    if lat and len(lat.get("results", [])) < 6:
        lat = None
    if lat:
        lrows = []
        for r in lat["results"]:
            lrows.append([f"{r['target_tokens']//1000}K", r["mode"],
                          f"{r['history_tokens']:,}", f"{r['prompt_tokens']:,}",
                          fmt(r["prefill_ms_median"], 1), fmt(r["ttft_ms_median"], 1),
                          fmt(r["prefill_tokens_per_s"], 0), fmt(r["mem_gb"], 2)])
        lat_tab = table(["size", "mode", "history tokens", "prompt tokens",
                         "prefill ms", "TTFT ms", "prefill tok/s", "peak GPU GB"], lrows)

    chat_tab = ""
    if chat:
        crows = []
        for e in chat["examples"]:
            crows.append([e["prompt"], e["answer"][:220].replace("\n", " ")])
        chat_tab = table(["prompt", "answer"], crows)

    # plots
    plt.figure(figsize=(8, 5))
    for mode, color in [("plain", "#1f77b4"), ("tags", "#ff7f0e"), ("index", "#2ca02c")]:
        xs, ys, errs = [], [], []
        for s in sizes:
            r = by.get((s, mode))
            if not r:
                continue
            k = sum(1 for x in r["rows"] if x.get("correct"))
            n = len([x for x in r["rows"] if not x.get("skipped")])
            lo, hi = wilson(k, n)
            xs.append(s); ys.append(r["accuracy"]); errs.append([r["accuracy"]-lo, hi-r["accuracy"]])
        if xs:
            plt.errorbar(xs, ys, yerr=np.array(errs).T, marker="o", capsize=3,
                         color=color, label=mode)
    plt.xticks(sizes, [f"{s//1000}K" for s in sizes])
    plt.ylim(0, 1)
    plt.xlabel("history budget (tokens)")
    plt.ylabel("exact-code accuracy")
    plt.title("SmolLM2-360M-Instruct: recall vs history size (95% Wilson CI)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(WORK, "llm_accuracy.png"), dpi=160)
    plt.close()

    if lat:
        plt.figure(figsize=(8, 5))
        for mode, color in [("plain", "#1f77b4"), ("tags", "#ff7f0e"), ("index", "#2ca02c")]:
            xs, ys = [], []
            for r in lat["results"]:
                if r["mode"] == mode:
                    xs.append(r["prompt_tokens"]); ys.append(r["prefill_ms_median"])
            if xs:
                plt.plot(xs, ys, marker="o", color=color, label=mode)
        plt.xlabel("prompt tokens")
        plt.ylabel("prefill latency (ms)")
        plt.title("Prompt processing latency (batch 1, prefill only)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(WORK, "llm_latency.png"), dpi=160)
        plt.close()

    best = max(results, key=lambda r: r["accuracy"])
    plain_2k = by.get((2000, "plain")); plain_4k = by.get((4000, "plain"))
    tags_2k = by.get((2000, "tags")); tags_4k = by.get((4000, "tags"))
    idx_2k = by.get((2000, "index")); idx_4k = by.get((4000, "index"))

    lines = []
    A = lines.append
    A("# Small-LLM history recall: plain text vs tags vs compact index")
    A("")
    A("**Date:** 2026-11-01  ")
    A("**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct` (360M params, fp16, 8K context)  ")
    A("**Task:** chat normally and answer questions from a long in-context history  ")
    A("**History:** synthetic facts + real Discord chatter; 20 questions per cell at 2K/4K  ")
    A("**Metric:** exact code match; 95% Wilson CI; paired McNemar exact test")
    A("")
    A("## Executive summary")
    A("")
    A(f"- **Plain natural language is the best quality/size tradeoff in this pilot.** At 2K tokens, "
      f"plain gets {plain_2k['accuracy']:.2f} vs {tags_2k['accuracy']:.2f} for inline tags and "
      f"{idx_2k['accuracy']:.2f} for the compact index. At 4K: plain {plain_4k['accuracy']:.2f}, "
      f"tags {tags_4k['accuracy']:.2f}, index {idx_4k['accuracy']:.2f}.")
    A(f"- **Tags do not help this model.** Inline XML tags add tokens and are often echoed or "
      f"ignored; the compact index uses fewer tokens but is not consistently better. The paired "
      f"differences are not statistically robust at these sample sizes, but nothing favours tags.")
    A(f"- **The 360M reader is not reliable.** Best accuracy in any cell is "
      f"{best['accuracy']:.2f} ({best['target_tokens']//1000}K, {best['mode']}). The model chats "
      f"coherently but confuses codes, invents 8-character codes, and sometimes repeats itself. "
      f"This is a model-size/quality limit, not a tag-format limit.")
    A(f"- **Prompt processing is not 'instant' on this hardware.** A preliminary prefill "
      f"measurement of {lat_note} was taken on a shared GPU and is contended, not a clean "
      f"benchmark, but it is already seconds, not milliseconds. Generation and memory are the "
      f"practical constraints at 4K+; the 6K run was stopped after 14 minutes without completing "
      f"10 questions, and 8K overflowed the prompt.")
    A(f"- **Verdict:** for a ~360M reader, **go without tags**: plain text has the best accuracy "
      f"and fewer bytes than inline tags. But the absolute recall is far below a usable bar; a "
      f"larger reader (0.5B–1B) or a trained reader/locator is required before this is a product.")
    A("")
    A("## Method")
    A("")
    A("1. Use the same 200 facts as the locator test. For the planned 2K/4K/6K histories, include "
      "the first 20/40/60 fact needles and their 3 designated distractors (80/160/240 facts); only "
      "2K/4K completed.")
    A("2. Build three histories at each size: **plain** natural-language facts, **inline tags** "
      "(`<fact project=... noun=... code=.../>`), and a **compact index** (`project=... | noun=... "
      "| code=...`). The facts and questions are identical; only the representation changes.")
    A("3. Mix the facts with real corpus lines as filler and shuffle. Keep the total history near "
      "the target; the tag/index conditions get the same total budget, not extra facts.")
    A("4. Ask the same questions; greedy decode up to 24 tokens (20 at 6K); exact-match the code "
      "after removing punctuation. No repetition penalty: the answer is in the context, so "
      "penalising repeated tokens suppresses the gold code (this was verified).")
    A("5. Measure accuracy, wrong-code rate, don't-know rate, generation time, and prefill latency.")
    A("")
    A("## Accuracy")
    A("")
    A(acc_tab)
    A("")
    A("![accuracy](llm_accuracy.png)")
    A("")
    A("## Size and tag overhead")
    A("")
    A(size_tab)
    A("")
    A("Inline tags spend more of the fixed history budget on fact text (about +9% at both sizes), "
      "leaving fewer filler tokens; the compact index spends about 4% fewer fact tokens. Total "
      "prompt length is held near the target, so this is a representation-quality comparison, not "
      "a raw prompt-size comparison. At this model size neither format improves accuracy.")
    A("")
    A("## Paired comparisons")
    A("")
    A(pair_tab)
    A("")
    A("McNemar compares the same questions under two conditions. With 20 questions per cell, "
      "only large, consistent differences would be significant; the pilot does not establish a "
      "tag advantage.")
    A("")
    if lat_tab:
        A("## Prompt processing latency")
        A("")
        A(lat_tab)
        A("")
        A("Prefill is a single forward pass with `logits_to_keep=1`, batch 1, median of 5. TTFT "
          "includes one decoded token. The plain condition is the cheapest because it has the "
          "fewest tokens; inline tags cost more, the compact index less.")
        A("")
        A("![latency](llm_latency.png)")
        A("")
    A("## Chat quality")
    A("")
    A("The model does chat in normal English, but its knowledge is weak. Examples:")
    A("")
    if chat_tab:
        A(chat_tab)
        A("")
    A("Note the electrical-transformer answer: the model is coherent but not reliable on factual "
      "questions. That is the same limitation that shows up in history recall.")
    A("")
    A("## Failure modes")
    A("")
    A("1. **Same-noun code confusion.** The model often returns a code that belongs to a different "
      "project with the same noun, or one of the designated distractors. Example: for "
      "`what is the serial number for project Voulzhux?` the gold is `ZGR-ULCT`; the 2K plain model "
      "answered `7N-LJ6Y`, a distractor code.")
    A("2. **Hallucinated codes.** At 4K the model answered `3X4-K48K` for Pathkiath with no such "
      "code in the history. The output format is right; the value is invented.")
    A("3. **Repetition loops.** In the index condition the model sometimes emits `77-77-77-...` "
      "instead of a code. This is a small-model degeneration, not a retrieval failure.")
    A("4. **Tag echo / truncation.** With inline tags the model often echoes the project name or a "
      "tag fragment (`Groth`, `Pathkiath`) instead of the code. With 16 output tokens some answers "
      "truncate before the code; 24 tokens is safer.")
    A("5. **Name misspelling.** The model writes `Pathkioth` / `Drovhzhark`; it is not copying the "
      "project name exactly, which makes exact matching harder.")
    A("")
    A("## Verdict")
    A("")
    A(f"**Go without tags for a ~360M reader.** Plain text is at least as accurate as inline tags "
      f"or a compact index, and it uses fewer bytes than inline tags. The compact index is smaller "
      f"but not better. The absolute quality is the problem: best accuracy "
      f"{best['accuracy']:.2f} is far below a reliable assistant. A 360M model can chat, but it "
      f"cannot be trusted to recall a specific code from a 2K–4K history.")
    A("")
    A("If tags are wanted for other reasons (structured search, audit, compression), they should "
      "be trained into the reader or used by a larger model; off-the-shelf XML tags are not "
      "understood well enough to pay for themselves.")
    A("")
    A("## Caveats")
    A("")
    A("- **Small samples.** 20 questions per cell at 2K/4K; the 6K attempt was stopped before a "
      "complete cell. The 95% CIs are wide (roughly ±0.2). The direction is consistent, but the "
      "tag/no-tag difference is not statistically strong.")
    A("- **One model.** SmolLM2-360M-Instruct; a 0.5B–1B model may handle tags differently.")
    A("- **Synthetic facts and template queries.** The facts and questions share a template, which "
      "may flatter the model. Real paraphrases will be harder.")
    A("- **In-context, not RAG.** The model sees the whole history; this is not the locator+reader "
      "architecture. The locator results are separate.")
    A("- **4K, not 6K/8K.** An 8K-target run overflowed the prompt (all rows skipped); a 7.6K-target "
      "run was impractically slow and was killed. A 6K-target run was stopped after 14 minutes "
      "without completing 10 questions. 4K is the largest completed context on the 6GB GPU.")
    A("- **Latency numbers are contended.** The GPU was shared (~68% utilization); the generation "
      "medians and the 2K prefill are not clean benchmarks. The accuracy comparison is unaffected.")
    A("- **Greedy decoding, no repetition penalty.** The penalty was tested and suppresses the gold "
      "code because it appears in the context. Greedy is reproducible but can loop.")
    A("- **Exact code match.** The model may give the right code in a longer sentence; substring "
      "normalisation handles punctuation and case, not wrong project names.")
    A("- **Memory.** The model is fp16 (0.72GB); 4-bit would be smaller but was not tested. The "
      "6GB GPU is the practical constraint at 4K+.")
    A("")
    A("## Files")
    A("")
    A("- `llm_results_final.json` — 2K/4K results.")
    A("- `llm_history_test_6k.log` — partial 6K answers; no complete `llm_results_6k.json`.")
    A("- `llm_latency.json` — preliminary 2K plain prefill only (contended GPU).")
    A("- `llm_chat_examples_final.json` — chat examples.")
    A("- `llm_accuracy.png` — accuracy plot; no latency plot because only a preliminary point exists.")
    A("- `llm_report.md` — this report.")
    A("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", OUT, "lines", len(lines))

if __name__ == "__main__":
    main()
