import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = "/home/daniel/tnc/work"
OUT = os.path.join(WORK, "report.md")
R = json.load(open(os.path.join(WORK, "results.json")))
SN = json.load(open(os.path.join(WORK, "same_noun_analysis.json")))
TL = json.load(open(os.path.join(WORK, "token_locator_results.json")))
DRIFT = None
if os.path.exists(os.path.join(WORK, "drift_true.json")):
    DRIFT = json.load(open(os.path.join(WORK, "drift_true.json")))
QL = None
if os.path.exists(os.path.join(WORK, "query_latency.json")):
    QL = json.load(open(os.path.join(WORK, "query_latency.json")))

NKEYS = ["131072", "1048576", "10485760"]
NLAB = {"131072": "128K", "1048576": "1M", "10485760": "9.75M"}
NVAL = {"131072": 131072, "1048576": 1048576, "10485760": 9746560}
CONFIGS = ["flat_fp32", "flat_fp16", "flat_int8", "pq8_early", "pq8_late",
           "pq16_early", "pq16_late", "pq32_early", "pq32_late",
           "pq48_early", "pq48_late"]
COLORS = {
    "flat_fp32": "#111111", "flat_fp16": "#1f77b4", "flat_int8": "#2ca02c",
    "pq8_early": "#d62728", "pq8_late": "#d62728",
    "pq16_early": "#ff7f0e", "pq16_late": "#ff7f0e",
    "pq32_early": "#9467bd", "pq32_late": "#9467bd",
    "pq48_early": "#8c564b", "pq48_late": "#8c564b",
}
STYLES = {c: ("-" if "late" not in c else "--") for c in CONFIGS}

def cfg(n, name):
    return R["stores"][n]["configs"][name]

def fmt(x, nd=3):
    return f"{x:.{nd}f}"

def table(headers, rows, align=None):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)

def plot_recall(k):
    plt.figure(figsize=(10, 6))
    for name in CONFIGS:
        ys = [cfg(n, name)[f"recall@{k}"] for n in NKEYS]
        plt.plot([NVAL[n] for n in NKEYS], ys, marker="o", markersize=4,
                 color=COLORS[name], linestyle=STYLES[name], linewidth=1.4,
                 label=name, alpha=0.9)
    if k == 10:
        plt.axhline(0.95, color="red", linestyle=":", linewidth=1, label="0.95 bar")
    plt.xscale("log")
    plt.xticks([NVAL[n] for n in NKEYS], [NLAB[n] for n in NKEYS])
    plt.ylim(0, 1.0)
    plt.xlabel("content tokens in store (log scale)")
    plt.ylabel(f"recall@{k}")
    plt.title(f"TNC-R locator recall@{k} vs store size")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    p = os.path.join(WORK, f"recall_at{k}.png")
    plt.savefig(p, dpi=160)
    plt.close()
    return p

def plot_memory():
    plt.figure(figsize=(9, 6))
    for name in CONFIGS:
        x = cfg("10485760", name)["total_bytes_per_token"]
        y = cfg("10485760", name)["recall@10"]
        plt.scatter([x], [y], color=COLORS[name], marker="o", s=45)
        plt.annotate(name, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    plt.axvline(16, color="red", linestyle=":", linewidth=1)
    plt.xlabel("total bytes per content token (largest store)")
    plt.ylabel("recall@10 at 9.75M tokens")
    plt.title("Memory vs recall@10 at the largest store")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(WORK, "memory_vs_recall.png")
    plt.savefig(p, dpi=160)
    plt.close()
    return p

def plot_latency():
    names = ["flat_fp32", "flat_fp16", "flat_int8", "pq8_early", "pq48_early"]
    plt.figure(figsize=(9, 6))
    for name in names:
        ys = [cfg(n, name)["latency"]["batch1_median_ms"] for n in NKEYS]
        plt.plot([NVAL[n] for n in NKEYS], ys, marker="o", color=COLORS[name],
                 linestyle=STYLES[name], label=name)
    plt.xscale("log"); plt.yscale("log")
    plt.xticks([NVAL[n] for n in NKEYS], [NLAB[n] for n in NKEYS])
    plt.xlabel("content tokens in store (log scale)")
    plt.ylabel("batch-1 retrieval latency (ms, log scale)")
    plt.title("Retrieval-only latency (no query embedding)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(WORK, "latency_batch1.png")
    plt.savefig(p, dpi=160)
    plt.close()
    return p

def plot_drift():
    m = [8, 16, 32, 48]
    early = [cfg("10485760", f"pq{x}_early")["recall@10"] for x in m]
    late = [cfg("10485760", f"pq{x}_late")["recall@10"] for x in m]
    x = np.arange(len(m)); w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, early, w, label="early (8K train)", color="#1f77b4")
    plt.bar(x + w/2, late, w, label="late (32K random train)", color="#ff7f0e")
    plt.xticks(x, [f"PQ{m_}" for m_ in m])
    plt.ylabel("recall@10 at 9.75M tokens")
    plt.ylim(0, 0.3)
    plt.title("PQ codebook training sample: existing early/late")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(WORK, "drift_max.png")
    plt.savefig(p, dpi=160)
    plt.close()
    return p

def plot_true_drift():
    if not DRIFT or "10485760" not in DRIFT.get("stores", {}):
        return None
    m = [8, 16, 32, 48]
    first8 = [DRIFT["stores"]["10485760"]["configs"][f"pq{x}_first_8192"]["recall@10"] for x in m]
    last8 = [DRIFT["stores"]["10485760"]["configs"][f"pq{x}_last_8192"]["recall@10"] for x in m]
    first32 = [DRIFT["stores"]["10485760"]["configs"][f"pq{x}_first_32768"]["recall@10"] for x in m]
    last32 = [DRIFT["stores"]["10485760"]["configs"][f"pq{x}_last_32768"]["recall@10"] for x in m]
    x = np.arange(len(m)); w = 0.2
    plt.figure(figsize=(9, 5))
    plt.bar(x - 1.5*w, first8, w, label="first 8K", color="#1f77b4")
    plt.bar(x - 0.5*w, last8, w, label="last 8K", color="#aec7e8")
    plt.bar(x + 0.5*w, first32, w, label="first 32K", color="#ff7f0e")
    plt.bar(x + 1.5*w, last32, w, label="last 32K", color="#ffbb78")
    plt.xticks(x, [f"PQ{m_}" for m_ in m])
    plt.ylabel("recall@10 at 9.75M tokens")
    plt.ylim(0, 0.3)
    plt.title("PQ codebook training sample: true first vs last")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    p = os.path.join(WORK, "drift_true.png")
    plt.savefig(p, dpi=160)
    plt.close()
    return p

def plot_same_noun():
    x = np.arange(len(NKEYS)); w = 0.2
    sn_top1 = [SN[n]["same_noun_top1_rate"] for n in NKEYS]
    sn_top10 = [SN[n]["same_noun_top10_rate"] for n in NKEYS]
    af_top1 = [SN[n]["any_fact_top1_rate"] for n in NKEYS]
    af_top10 = [SN[n]["any_fact_top10_rate"] for n in NKEYS]
    plt.figure(figsize=(8, 5))
    plt.bar(x - 1.5*w, sn_top1, w, label="same-noun top-1")
    plt.bar(x - 0.5*w, sn_top10, w, label="same-noun top-10")
    plt.bar(x + 0.5*w, af_top1, w, label="any-fact top-1")
    plt.bar(x + 1.5*w, af_top10, w, label="any-fact top-10")
    plt.xticks(x, [NLAB[n] for n in NKEYS])
    plt.ylim(0, 1.05)
    plt.ylabel("rate over 200 queries")
    plt.title("What the flat fp32 retriever returns")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    p = os.path.join(WORK, "same_noun.png")
    plt.savefig(p, dpi=160)
    plt.close()
    return p

def main():
    for k in [1, 10, 50]:
        plot_recall(k)
    plot_memory(); plot_latency(); plot_drift(); plot_true_drift(); plot_same_noun()

    t = R["tokens"]
    mem_rows = []
    content_tokens_max = R["stores"]["10485760"]["content_tokens"]
    for name in CONFIGS:
        c = cfg("10485760", name)
        idx_bpt = c["bytes_per_chunk"] / 128.0
        cb_bpt = c["codebook_bytes"] / content_tokens_max
        loc_bpt = idx_bpt + cb_bpt
        total = 4.0 + loc_bpt
        mem_rows.append([name, fmt(idx_bpt, 4), fmt(cb_bpt, 4), fmt(loc_bpt, 4),
                         "4.000", fmt(total, 4), fmt(16 - total, 4)])
    mem_max = table(["config", "index payload B/token", "codebook B/token",
                     "locator B/token", "token IDs + flags", "total B/token",
                     "headroom vs 16"], mem_rows)

    total_rows = []
    for name in CONFIGS:
        vals = [cfg(n, name)["total_bytes_per_token"] for n in NKEYS]
        total_rows.append([name] + [fmt(v, 3) for v in vals])
    total_tab = table(["config", "128K", "1M", "9.75M"], total_rows)

    rec_tables = {}
    for k in [1, 10, 50]:
        rows = []
        best = {n: max(cfg(n, name)[f"recall@{k}"] for name in CONFIGS) for n in NKEYS}
        for name in CONFIGS:
            vals = []
            for n in NKEYS:
                v = cfg(n, name)[f"recall@{k}"]
                s = fmt(v, 3)
                if abs(v - best[n]) < 1e-12:
                    s = "**" + s + "**"
                vals.append(s)
            rows.append([name] + vals)
        rec_tables[k] = table(["config", "128K", "1M", "9.75M"], rows)

    drift_rows = []
    for m in [8, 16, 32, 48]:
        e = cfg("10485760", f"pq{m}_early"); l = cfg("10485760", f"pq{m}_late")
        drift_rows.append([f"PQ{m}"] + [fmt(e[f"recall@{k}"], 3) for k in [1, 10, 50]] +
                          [fmt(l[f"recall@{k}"], 3) for k in [1, 10, 50]] +
                          [fmt(l[f"recall@{k}"] - e[f"recall@{k}"], 3) for k in [1, 10, 50]])
    drift_tab = table(["config", "early r@1", "early r@10", "early r@50",
                       "late r@1", "late r@10", "late r@50",
                       "Δ r@1", "Δ r@10", "Δ r@50"], drift_rows)

    true_drift_tab = ""
    if DRIFT and "10485760" in DRIFT.get("stores", {}):
        rows = []
        for m in [8, 16, 32, 48]:
            row = [f"PQ{m}"]
            for sample in [8192, 32768]:
                for side in ["first", "last"]:
                    c = DRIFT["stores"]["10485760"]["configs"][f"pq{m}_{side}_{sample}"]
                    row.append(fmt(c["recall@10"], 3))
            rows.append(row)
        true_drift_tab = table(["config", "first 8K", "last 8K", "first 32K", "last 32K"], rows)

    lat_rows = []
    for name in CONFIGS:
        c = cfg("10485760", name); l = c["latency"]
        lat_rows.append([name, fmt(l["batch1_median_ms"], 2), fmt(l["batch1_p95_ms"], 2),
                         fmt(l["batch200_median_ms"], 2), fmt(l["batch200_qps"], 0),
                         fmt(c["index_payload_bytes"] / 1e6, 1), fmt(c["total_bytes_per_token"], 3)])
    lat_tab = table(["config", "batch1 median ms", "batch1 p95 ms",
                     "batch200 median ms", "batch200 qps", "index payload MB", "total B/token"], lat_rows)

    query_tab = ""
    e2e_tab = ""
    if QL:
        qrows = [
            ["1", fmt(QL["batch1_median_ms"], 2), fmt(QL["batch1_p95_ms"], 2), fmt(QL["batch1_qps"], 1)],
            ["200", fmt(QL["batch200_median_ms"], 2), "-", fmt(QL["batch200_qps"], 0)],
        ]
        query_tab = table(["batch", "median ms", "p95 ms", "queries/s"], qrows)
        erows = []
        for name in ["flat_fp32", "flat_fp16", "flat_int8", "pq8_early", "pq48_early"]:
            c = cfg("10485760", name); l = c["latency"]
            e2e_b1 = l["batch1_median_ms"] + QL["batch1_median_ms"]
            e2e_b200 = l["batch200_median_ms"] + QL["batch200_median_ms"]
            erows.append([name, fmt(l["batch1_median_ms"], 2), fmt(QL["batch1_median_ms"], 2),
                          fmt(e2e_b1, 2), fmt(l["batch200_median_ms"], 2),
                          fmt(QL["batch200_median_ms"], 2), fmt(e2e_b200, 2),
                          fmt(e2e_b200 / 200 * 1000, 0)])
        e2e_tab = table(["config", "retr b1 ms", "embed b1 ms", "e2e b1 ms",
                         "retr b200 ms", "embed b200 ms", "e2e b200 ms", "e2e b200 µs/query"], erows)

    ext_rows = []
    for name in CONFIGS:
        e100 = R["latency_extrapolation"]["targets"]["100000000"][name]
        e500 = R["latency_extrapolation"]["targets"]["500000000"][name]
        ext_rows.append([name, fmt(e100["batch1_median_ms"], 1), fmt(e100["batch200_median_ms"], 1),
                         fmt(e500["batch1_median_ms"], 1), fmt(e500["batch200_median_ms"], 1),
                         fmt(e500["index_payload_bytes"] / 1e9, 2)])
    ext_tab = table(["config", "100M batch1 ms", "100M batch200 ms",
                     "500M batch1 ms", "500M batch200 ms", "500M index GB"], ext_rows)

    sn_rows = []
    for n in NKEYS:
        s = SN[n]
        c = cfg(n, "flat_fp32")
        sn_rows.append([NLAB[n], fmt(c["recall@1"], 3), fmt(c["recall@10"], 3),
                        fmt(s["same_noun_top1_rate"], 3), fmt(s["same_noun_top10_rate"], 3),
                        fmt(s["any_fact_top1_rate"], 3), fmt(s["any_fact_top10_rate"], 3),
                        fmt(s["median_rank"], 0)])
    sn_tab = table(["store", "r@1", "r@10", "same-noun top1", "same-noun top10",
                    "any-fact top1", "any-fact top10", "median rank"], sn_rows)

    best = cfg("10485760", "flat_int8")
    r10_128 = cfg("131072", "flat_int8")["recall@10"]
    r10_1m = cfg("1048576", "flat_int8")["recall@10"]
    loss_128 = r10_128 - best["recall@10"]
    loss_1m = r10_1m - best["recall@10"]

    lines = []
    A = lines.append
    A("# TNC-R locator recall at 128K / 1M / 9.75M content tokens")
    A("")
    A("**Date:** 2026-11-01  ")
    A("**Corpus:** 603,531 Discord messages across 28 channels  ")
    A("**Embedder:** `BAAI/bge-small-en-v1.5` (384-d, fp16)  ")
    A("**Store:** 128-content-token chunks; 200 needle facts + 600 hard distractors  ")
    A("**Metric:** recall@1/10/50 over 200 queries; L2 on normalised embeddings (cosine ranking)")
    A("")
    A("## Executive summary")
    A("")
    A(f"- The best 16 B/token configuration at the largest store is **flat int8** with "
      f"**recall@10 {best['recall@10']:.3f}** (recall@1 {best['recall@1']:.3f}, "
      f"recall@50 {best['recall@50']:.3f}) at **{best['total_bytes_per_token']:.3f} B/token**. "
      f"It matches fp32/fp16 recall exactly, so the 16 B/token budget is not the binding constraint.")
    A(f"- Recall@10 falls from **{r10_128:.3f}** at 128K to **{best['recall@10']:.3f}** at 9.75M "
      f"({loss_128*100:.1f} points), and from **{r10_1m:.3f}** at 1M to **{best['recall@10']:.3f}** "
      f"({loss_1m*100:.1f} points). The curve is flat, not a cliff.")
    A(f"- PQ trades quality for bytes. At 9.75M, PQ48 reaches recall@10 "
      f"{cfg('10485760','pq48_early')['recall@10']:.3f} at "
      f"{cfg('10485760','pq48_early')['total_bytes_per_token']:.3f} B/token; PQ8 collapses to "
      f"{cfg('10485760','pq8_early')['recall@10']:.3f}. At 128K, flat int8 "
      f"({cfg('131072','flat_int8')['total_bytes_per_token']:.2f} B/token, "
      f"{cfg('131072','flat_int8')['recall@10']:.3f}) dominates PQ48 "
      f"({cfg('131072','pq48_early')['total_bytes_per_token']:.2f} B/token, "
      f"{cfg('131072','pq48_early')['recall@10']:.3f}).")
    A(f"- The dominant failure is **same-noun confusion**, not generic chatter alone. At 9.75M the "
      f"top-1 result is a fact with the same noun but a different project in "
      f"**{SN['10485760']['same_noun_top1_rate']*100:.1f}%** of queries; a same-noun fact is in the "
      f"top 10 for **{SN['10485760']['same_noun_top10_rate']*100:.1f}%**. The three designated "
      f"distractors are the top-1 only {cfg('10485760','flat_fp32')['distractor_top1_rate']*100:.1f}% "
      f"of the time; the whole ~132-fact same-noun pool is the real distractor set.")
    A(f"- A crude per-token window test (32 mean-pooled BGE windows/chunk, PQ m=4, ColBERT-lite "
      f"max-sim, {TL['total_bytes_per_token']:.2f} B/token) is much worse: recall@10 "
      f"**{TL['recall@10']:.3f}**, median gold rank **{TL['median_rank']}**. This does not rule out a "
      f"trained token-level locator, but off-the-shelf token states are not a drop-in fix.")
    if QL:
        A(f"- Query embedding is a fixed **{QL['batch1_median_ms']:.1f} ms** at batch 1 and "
          f"**{QL['batch200_median_ms']:.1f} ms per 200 queries**. End-to-end at 9.75M, flat int8 is "
          f"**{cfg('10485760','flat_int8')['latency']['batch1_median_ms'] + QL['batch1_median_ms']:.1f} ms** "
          f"at batch 1 and "
          f"**{cfg('10485760','flat_int8')['latency']['batch200_median_ms'] + QL['batch200_median_ms']:.0f} ms** "
          f"per 200. The query encoder, not the index, dominates single-query latency.")
    A(f"- **Verdict:** under the pre-decided rules the dense-only result is **inconclusive** — not "
      f"viable ({best['recall@10']:.3f} is far below 0.95) and not \"in trouble\" by the 15-point rule "
      f"(1M→9.75M loss is {loss_1m*100:.1f} points). The measured floor is far below viability. "
      f"A lexical/hybrid follow-up changes the conclusion for named-entity queries: word BM25 + "
      f"char 2–3-gram router gets **0.995 exact-name / 0.650 typo r@10 at 44.32 B/token total** "
      f"(see `hybrid_report.md`). The chunk-level BGE locator should not be shipped.")
    A("")
    A("## Method")
    A("")
    A("1. Tokenise the corpus with GPT-2; mark tokens that begin a blank run (0–3+ blanks).")
    A("2. Pack **128 content tokens** per chunk, dropping the 1,804-token partial tail; shuffle with "
      "seed 20261030.")
    A("3. Embed chunks with `bge-small-en-v1.5` (fp16, max sequence 256), L2-normalise, store fp32.")
    A("4. Inject 200 synthetic facts (`Note to self: the <noun> for project <name> is <code>.`) plus "
      "3 hard distractors per fact into 800 chunks per size. Queries use the BGE retrieval prefix.")
    A("5. Build flat fp32/fp16/SQ8 indexes and PQ m=8/16/32/48 indexes. The existing \"early\" "
      "codebook trains on the first 8,192 vectors; \"late\" on a 32,768-vector random sample. "
      "A supplementary true first-vs-last test is included below.")
    A("6. Measure recall@1/10/50, same-noun top-1/top-10, and retrieval-only latency (batch 1 and 200).")
    A("")
    A("## Token accounting")
    A("")
    A(table(["quantity", "value"], [
        ["messages", f"{t['messages']:,}"],
        ["channels", f"{t['channels']:,}"],
        ["raw GPT-2 tokens", f"{t['raw_tokens']:,}"],
        ["blank tokens", f"{t['blank_tokens']:,}"],
        ["blank fraction", f"{t['blank_fraction']*100:.2f}%"],
        ["content tokens", f"{t['content_tokens']:,}"],
        ["content tokens in full 128-token chunks", f"{t['content_tokens_in_full_chunks']:,}"],
        ["content tokens dropped in partial tails", f"{t['content_tokens_dropped_in_partial_chunks']:,}"],
        ["chunks", f"{t['n_chunks']:,}"],
        ["blank-run flag distribution (0/1/2/3+)", " / ".join(f"{t['blank_flag_distribution_content_tokens'][k]:,}" for k in ["0","1","2","3"])],
    ]))
    A("")
    A(f"The 10M request is capped by the corpus at **{NVAL['10485760']:,} content tokens** "
      f"({NVAL['10485760']/10_000_000*100:.1f}% of 10M). The 128K and 1M stores use exactly "
      f"{NVAL['131072']:,} and {NVAL['1048576']:,} content tokens.")
    A("")
    A("## Memory per content token")
    A("")
    A("Base cost is **4 B/token**: 3 B token ID + 1 B blank-run flag. The locator bytes are the "
      "index payload plus its persistent codebook, amortised over the store. Headroom is against "
      "the 16 B/token budget.")
    A("")
    A("### At the largest store (9,746,560 content tokens)")
    A("")
    A(mem_max)
    A("")
    A("### Total bytes per content token at each size")
    A("")
    A(total_tab)
    A("")
    A("The PQ codebook is 393,216 B per index. It costs **3.0 B/token at 128K**, **0.375 B/token "
      "at 1M**, and **0.040 B/token at 9.75M**; this is why PQ is only competitive at the largest "
      "store. Flat int8's 3,072 B min/max parameters are negligible at every size.")
    A("")
    A("## Recall")
    A("")
    A("### Recall@1")
    A("")
    A(rec_tables[1])
    A("")
    A("### Recall@10")
    A("")
    A(rec_tables[10])
    A("")
    A("### Recall@50")
    A("")
    A(rec_tables[50])
    A("")
    A("![recall@1](recall_at1.png)")
    A("")
    A("![recall@10](recall_at10.png)")
    A("")
    A("![recall@50](recall_at50.png)")
    A("")
    A("![memory vs recall](memory_vs_recall.png)")
    A("")
    A("## Early vs late codebooks")
    A("")
    A("The existing \"early/late\" pair is **not a temporal split**: the store is shuffled, "
      "\"early\" trains on the first 8,192 vectors, and \"late\" trains on a 32,768-vector random "
      "sample. At 128K and 1M the recalls are identical to three decimals; at 9.75M the differences "
      "are small and mostly negative for the larger random sample.")
    A("")
    A(drift_tab)
    A("")
    A("![early/late](drift_max.png)")
    A("")
    if true_drift_tab:
        A("### Supplementary true first-vs-last codebook test")
        A("")
        A("To answer the literal first-vs-last question, the codebook was retrained on the first and "
          "last 8,192 and 32,768 vectors of the shuffled store at 9.75M tokens. This is still not a "
          "temporal test (the store is shuffled), but it isolates which sample is used.")
        A("")
        A(true_drift_tab)
        A("")
        A("![true first/last](drift_true.png)")
        A("")
    A("## Latency")
    A("")
    A("Retrieval-only, measured on the largest store with 8 FAISS threads. Query embedding, I/O, "
      "and reranking are excluded. Batch 1 is the median of 200 queries; batch 200 is the median of "
      "5 repeats.")
    A("")
    A(lat_tab)
    A("")
    A("![batch-1 latency](latency_batch1.png)")
    A("")
    if query_tab:
        A("### Query embedding latency")
        A("")
        A("Same `bge-small-en-v1.5` model, fp16, max sequence 256, measured on the query set. "
          "Batch 1 includes tokenisation and Python overhead; batch 200 is the median of 5 repeats.")
        A("")
        A(query_tab)
        A("")
        A("### End-to-end latency at the largest store")
        A("")
        A("Sum of medians (query embedding + index search), not the median of sums. Batch 200 is "
          "for 200 queries; the last column is per query.")
        A("")
        A(e2e_tab)
        A("")
        A("At batch 1 the query embedder dominates every index: 12.4 ms of a 16.6 ms end-to-end "
          "flat-int8 query. At batch 200 the index dominates for flat, while the embedder is 30 ms "
          "of 182 ms. The 16 B/token budget is not the latency problem; the query encoder is the "
          "fixed cost.")
        A("")
    A("### Extrapolation to 100M and 500M content tokens")
    A("")
    A("Linear in store size (chunks), marked **extrapolation**. Flat indexes scan every vector, so "
      "this is optimistic for 500M; PQ also scans but with much smaller codes.")
    A("")
    A(ext_tab)
    A("")
    A("## Same-noun failure analysis")
    A("")
    A(sn_tab)
    A("")
    A("![same-noun rates](same_noun.png)")
    A("")
    A("A same-noun fact is in the top 10 for **every** query at every size. The retriever is not "
      "failing to find the noun; it is failing to disambiguate the project within the noun. The "
      "designated three distractors are a small fraction of that pool: there are ~132 facts per "
      "noun, so a query competes against ~131 same-noun alternatives, not 3.")
    A("")
    A("## Token-window test")
    A("")
    A(table(["metric", "value"], [
        ["method", "32 mean-pooled BGE token windows/chunk; PQ m=4 (4 B/window); ColBERT-lite max-sim"],
        ["bytes/token", f"{TL['total_bytes_per_token']:.3f}"],
        ["recall@1", f"{TL['recall@1']:.3f}"],
        ["recall@10", f"{TL['recall@10']:.3f}"],
        ["recall@50", f"{TL['recall@50']:.3f}"],
        ["same-noun top1", f"{TL['same_noun_top1_rate']:.3f}"],
        ["same-noun top10", f"{TL['same_noun_top10_rate']:.3f}"],
        ["median gold rank", f"{TL['median_rank']}"],
        ["search time", f"{TL['search_seconds']:.1f} s for 200 queries"],
    ]))
    A("")
    A("This is a **negative result for the crude method**, not for the per-token idea. Mean-pooling "
      "BGE states into 32 windows and max-simming query tokens loses too much information; a trained "
      "token-level objective is untested.")
    A("")
    A("## Failure modes (ranked)")
    A("")
    A("1. **Same-noun, wrong-project confusion (dominant).** The top-1 chunk often contains a fact "
      "for a different project with the same noun. Examples:")
    A("")
    A(table(["query", "gold rank", "top-1 excerpt", "why"], [
        ["invoice number for project Pathkiath", "31",
         "NYT link chunk ending `invoice number for project Drou is PAP-UJGW`",
         "same noun, wrong project; URL chunk is a high-similarity hub"],
        ["serial number for project Voulzhux", "42",
         "model-list chunk ending `serial number for project Skouthbrith is UHL-NAMP`",
         "same noun, wrong project; hub chunk"],
        ["tracking ID for project Droulzhark", ">50",
         "URL-encoded chunk ending `tracking ID for project Grourkdrark is 593-PL4T`",
         "same noun, wrong project; gold not retrieved"],
    ]))
    A("")
    A("2. **Chunk-level dilution.** The fact is appended to the end of a 128-token chunk of real "
      "chatter. The chunk embedding is an average over the whole chunk, so the fact is a small part "
      "of the vector. Diagnostic fact-only matching reaches 0.995 recall@1, so the embedder can match "
      "the fact sentence; the loss is in pooling it with 128 tokens of unrelated text.")
    A("")
    A("3. **Hub chunks.** A handful of chunks (model lists, URL-heavy messages, code blocks) are "
      "similar to many queries and also contain a same-noun fact. One hub can be the top-1 for "
      "several different projects. This is a corpus-structure problem as much as an embedding "
      "problem.")
    A("")
    A("4. **Quantisation loss (PQ only).** PQ48 can miss a fact that flat finds. Example: "
      "`access code for project Groth` is flat rank 3 but PQ48 rank >50; the top-1 is a chatter "
      "chunk with a passphrase fact. PQ8 is unusable at this store size (recall@10 0.010).")
    A("")
    A("5. **Gold fact buried or absent from top 50.** Some gold chunks never enter the top 50 "
      "(rank 0 in the failure dump), even though the fact-only diagnostic can match the sentence. "
      "This is the extreme end of dilution + same-noun competition.")
    A("")
    A("## Verdict")
    A("")
    A("Pre-decided rules:")
    A("")
    A("- **Viable:** recall@10 ≥ 0.95 at the largest store and loss from 128K < 5 points.")
    A("- **In trouble:** loss from 1M to largest > 15 points; extrapolate collapse.")
    A("- **Inconclusive:** fails viability but does not meet the collapse rule; say why.")
    A("")
    A(f"Best under-budget config: **flat int8**, recall@10 **{best['recall@10']:.3f}** at "
      f"{best['total_bytes_per_token']:.3f} B/token. Loss 128K→9.75M is **{loss_128*100:.1f} points**; "
      f"loss 1M→9.75M is **{loss_1m*100:.1f} points**.")
    A("")
    A("**Result: inconclusive under the letter of the rules, but not viable in practice.** The "
      "measured floor is 0.405, less than half the 0.95 bar. The 16 B/token budget is not what is "
      "stopping the design; flat int8 uses 7.0 B/token and matches fp32. The stopping problem is "
      "chunk-level discrimination among same-noun facts. The dense-only design should not be shipped.")
    A("")
    A("## Follow-up: lexical/hybrid retrieval at 64 B/token")
    A("")
    A("The dense-only result is not the end of the story. On the same 9.75M store, a word BM25 index "
      "(3.09 B/token) gets **r@10 0.995** when the query names the project; a hashed character "
      "2–3-gram index (37.23 B/token) gets **r@10 0.650** when the project name has a one-character "
      "typo. A router that uses word BM25 when the entity token is present and the character index "
      "otherwise gets **0.995 exact-name / 0.650 typo r@10 at 44.32 B/token total**, under the "
      "64 B/token allowance. Without the project name the query is ambiguous (word BM25 r@10 0.045), so entity "
      "resolution or conversation state is still required. At 100M tokens (781,250 chunks, repeated "
      "filler), the exact path stays at r@10 1.000 and the typo fallback at 0.560 for 43.63 B/token "
      "total; see `hybrid_100m_report.md`. Full 9.75M results: `hybrid_report.md`.")
    A("")
    A("## Caveats and assumptions")
    A("")
    A("- **Synthetic facts.** The 800 injected facts add ~16.4K tokens at 128K (12.5% over the "
      "nominal store), ~1.6% at 1M, and ~0.17% at 9.75M. The memory table counts the original "
      "content tokens, not the injected fact tokens. Counting them would raise bytes/token by the "
      "same percentage; the ranking of configs is unchanged.")
    A("- **Token IDs.** The required 3 B/token is conservative for this corpus: GPT-2's 50,257-token "
      "vocab fits in 2 B. Using 2 B would free 1 B/token of headroom. The task specified 3 B, so all "
      "tables use 3 B.")
    A("- **Blank flag.** 1 B/token stores a 2-bit flag in a byte. Packing 4 flags/byte would cost "
      "0.25 B/token and free 0.75 B/token. The task specified 1 B.")
    A("- **Partial tail.** 1,804 content tokens (0.018%) are dropped. Negligible.")
    A("- **Shuffle.** The corpus is shuffled with seed 20261030. \"Early/late\" is therefore not "
      "chronological drift. The supplementary first-vs-last test is also on the shuffled order; a "
      "true temporal test needs an unshuffled corpus.")
    A("- **Query prefix.** The BGE retrieval prefix is used. A no-prefix diagnostic was slightly "
      "better at recall@1 but worse at recall@10/50; the main tables use the prefix.")
    A("- **Fact-only diagnostic.** Matching the query to the fact sentence alone reaches 0.995 "
      "recall@1, so the embedder is capable; chunk pooling and same-noun competition cause the loss.")
    A("- **Latency.** Retrieval-only; no query embedding, I/O, reranking, or network. Extrapolation "
      "is linear and marked; flat indexes at 500M are optimistic.")
    A("- **Token-window test.** Crude mean-pooled BGE windows + ColBERT-lite max-sim; not a trained "
      "token-level locator.")
    A("- **PQ codebook.** 393,216 B per index, counted as persistent at every store size. The "
      "codebook is trained on 8K/32K vectors, not the full store.")
    A("- **Corpus cap.** The 10M request is 9,746,560 content tokens (97.5% of 10M).")
    A("")
    A("## What to test next")
    A("")
    A("1. **Train the locator with same-noun hard negatives.** This is the direct fix for the "
      "dominant failure. Fine-tune BGE (or a small dual encoder) contrastively with in-batch and "
      "same-noun negatives; measure the same 200-query set.")
    A("2. **Reduce chunk dilution.** Try 32/64-token chunks or fact-aware pooling at the same "
      "16 B/token budget. The fact-only diagnostic says the signal exists; the chunk average is "
      "washing it out.")
    A("3. **Train a token-level locator.** The crude window test is negative, but a learned "
      "token/window objective (e.g., ColBERT-style) is the fair test of the per-token route.")
    A("4. **True temporal drift.** Re-run the codebook test on an unshuffled corpus, training on "
      "the first vs last chronological portions.")
    A("5. **Real questions.** Replace the template queries with paraphrases and multi-hop questions; "
      "the current queries share a template with the facts, which may flatter the embedder.")
    A("6. **Scale test.** At 100M/500M, use IVF/HNSW rather than flat; measure recall vs latency "
      "and memory. The linear flat extrapolation is not a deployment plan.")
    A("7. **Reader test.** A small instruct LLM with and without tags is a separate pilot; see "
      "`llm_report.md` when available.")
    A("")
    A("## Files")
    A("")
    A("- `results.json` — all recall, memory, latency, and failure-example data.")
    A("- `report.md` — this report.")
    A("- `recall_at1.png`, `recall_at10.png`, `recall_at50.png` — recall curves.")
    A("- `memory_vs_recall.png` — bytes/token vs recall@10 at 9.75M.")
    A("- `latency_batch1.png` — retrieval latency vs store size.")
    A("- `drift_max.png`, `drift_true.png` — codebook training-sample tests.")
    A("- `same_noun.png` — same-noun failure rates.")
    A("- `token_locator_results.json` — per-token window negative result.")
    A("- `query_latency.json` — query embedding latency.")
    A("- `drift_true.json` — supplementary first/last codebook results.")
    A("- `hybrid_report.md`, `hybrid_results.json`, `hybrid_quality.png` — lexical/hybrid follow-up.")
    A("- `hybrid_100m_report.md`, `hybrid_results_100m.json` — 100M-token scale test.")
    A("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", OUT, "lines", len(lines))

if __name__ == "__main__":
    main()
