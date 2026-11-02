import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = "/home/daniel/tnc/work"
H = json.load(open(os.path.join(WORK, "hybrid_results.json")))
R = json.load(open(os.path.join(WORK, "results.json")))
SN = json.load(open(os.path.join(WORK, "same_noun_analysis.json")))
C = H["conditions"]
dense = R["stores"]["10485760"]["configs"]["flat_int8"]

def pct(x):
    return f"{x:.3f}"

rows = [
    ["dense bge-small", pct(dense["recall@1"]), pct(dense["recall@10"]), pct(dense["recall@50"]), "—", "—", "—"],
    ["word BM25", pct(C["word_exact"]["recall@1"]), pct(C["word_exact"]["recall@10"]), pct(C["word_exact"]["recall@50"]),
     pct(C["word_typo"]["recall@1"]), pct(C["word_typo"]["recall@10"]), pct(C["word_typo"]["recall@50"])],
    ["char 2-3gram", pct(C["char_exact"]["recall@1"]), pct(C["char_exact"]["recall@10"]), pct(C["char_exact"]["recall@50"]),
     pct(C["char_typo"]["recall@1"]), pct(C["char_typo"]["recall@10"]), pct(C["char_typo"]["recall@50"])],
    ["RRF word+char", pct(C["rrf_exact"]["recall@1"]), pct(C["rrf_exact"]["recall@10"]), pct(C["rrf_exact"]["recall@50"]),
     pct(C["rrf_typo"]["recall@1"]), pct(C["rrf_typo"]["recall@10"]), pct(C["rrf_typo"]["recall@50"])],
    ["router", pct(C["route_exact"]["recall@1"]), pct(C["route_exact"]["recall@10"]), pct(C["route_exact"]["recall@50"]),
     pct(C["route_typo"]["recall@1"]), pct(C["route_typo"]["recall@10"]), pct(C["route_typo"]["recall@50"])],
]
tab = ["| retriever | exact r@1 | exact r@10 | exact r@50 | typo r@1 | typo r@10 | typo r@50 |",
       "|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    tab.append("| " + " | ".join(r) + " |")
tab = "\n".join(tab)

base_bytes = H["n_chunks"] * 128 * 4.0
loc_bytes = H['word_est_bytes'] + H['char_est_bytes']
store = f"""| component | MB | B/token |
|---|---:|---:|
| word BM25 | {H['word_est_bytes']/1e6:.1f} | {H['word_est_bytes_per_token']:.2f} |
| char 2-3gram TF-IDF | {H['char_est_bytes']/1e6:.1f} | {H['char_est_bytes_per_token']:.2f} |
| **locator combined** | **{loc_bytes/1e6:.1f}** | **{H['combined_est_bytes_per_token']:.2f}** |
| base token IDs + blank flag | {base_bytes/1e6:.1f} | 4.00 |
| **total without dense** | **{(loc_bytes+base_bytes)/1e6:.1f}** | **{H['combined_est_bytes_per_token']+4.0:.2f}** |
| + dense bge-small | 68.2 | 7.00 |
| **total with dense** | **{(loc_bytes+base_bytes+68.2e6)/1e6:.1f}** | **{H['combined_est_bytes_per_token']+4.0+7.0:.2f}** |
| budget | | 64.00 |"""

# plot
labels = ["word", "char", "RRF", "router"]
exact = [C["word_exact"]["recall@10"], C["char_exact"]["recall@10"],
         C["rrf_exact"]["recall@10"], C["route_exact"]["recall@10"]]
typo = [C["word_typo"]["recall@10"], C["char_typo"]["recall@10"],
        C["rrf_typo"]["recall@10"], C["route_typo"]["recall@10"]]
x = np.arange(len(labels)); w = 0.36
plt.figure(figsize=(8, 4.8))
plt.bar(x - w/2, exact, w, label="exact name", color="#1f77b4")
plt.bar(x + w/2, typo, w, label="one-char typo", color="#ff7f0e")
plt.axhline(dense["recall@10"], color="#d62728", linestyle="--", linewidth=1.5,
            label=f"dense bge-small exact ({dense['recall@10']:.3f})")
plt.xticks(x, labels)
plt.ylim(0, 1.05)
plt.ylabel("recall@10")
plt.title("9.75M tokens: lexical/hybrid recall")
plt.grid(True, axis="y", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(WORK, "hybrid_quality.png"), dpi=160)
plt.close()

lines = []
A = lines.append
A("# Lexical/hybrid follow-up: the locator is viable at 44 B/token")
A("")
A("**Store:** 76,145 chunks / 9,746,560 content tokens, 200 queries  ")
A("**Budget:** 64 B/token allowed; combined total is **44.32 B/token** (40.32 locator + 4 base)  ")
A("**Metric:** exact gold chunk in top-k; same-noun top-1 is a wrong-project fact")
A("")
A("## Executive summary")
A("")
A(f"- **Dense-only was the wrong tool.** `bge-small` flat int8 gets recall@10 "
  f"**{dense['recall@10']:.3f}** at 9.75M; the same store with word BM25 gets "
  f"**{C['word_exact']['recall@10']:.3f}**.")
A(f"- **Exact-name queries are essentially solved by lexical retrieval.** Word BM25: "
  f"r@1 **{C['word_exact']['recall@1']:.3f}**, r@10 **{C['word_exact']['recall@10']:.3f}**, "
  f"r@50 **{C['word_exact']['recall@50']:.3f}**, at **3.09 B/token**.")
A(f"- **A single-character typo breaks exact BM25** (r@10 **{C['word_typo']['recall@10']:.3f}**), "
  f"but a character 2–3-gram index recovers **{C['char_typo']['recall@10']:.3f}** at r@10.")
A(f"- **The router gets both:** exact **{C['route_exact']['recall@10']:.3f}**, typo "
  f"**{C['route_typo']['recall@10']:.3f}** at r@10, for **44.32 B/token total** — under the "
  f"64 B/token budget. Same-noun top-1 drops from **0.795** (dense) to **0.020** (router exact).")
A(f"- **Verdict:** the idea is possible **for queries that name the entity**. It is not solved for "
  f"pronouns, paraphrases without the name, or multi-turn references; those need a dense/query-rewrite "
  f"fallback. But the 0.405 ceiling was a dense-retrieval artifact, not a property of the task.")
A("")
A("## Results at 9.75M tokens")
A("")
A(tab)
A("")
A("![hybrid quality](hybrid_quality.png)")
A("")
A("The typo condition replaces the project name with a last-two-character swap "
  "(`Groth` → `Groht`) in every query. The gold fact is unchanged.")
A("")
A(f"**No-name sanity check:** with the project name removed (`what is the access code for the "
  f"project?`), word BM25 gets r@10 **{C['word_noname']['recall@10']:.3f}** and char n-gram gets "
  f"**{C['char_noname']['recall@10']:.3f}**; same-noun top-1 is **0.990**. The query is genuinely "
  f"ambiguous among ~136 same-noun facts. Lexical retrieval only works when the entity is named.")
A("")
A("## Storage")
A("")
A(store)
A("")
A("The char index estimate is `nnz × 8 B + 2^20 × 4 B` for a hashed CSR matrix; the word BM25 "
  "estimate is `postings × 5 B + vocabulary × 24 B`. Both are in-memory estimates, not compressed "
  "on-disk sizes. Even with dense bge-small added, the total is ~51 B/token, leaving ~13 B/token "
  "of headroom.")
A("")
A("## Method")
A("")
A("1. Build word BM25 over the modified chunks (`k1=1.2`, `b=0.75`, token pattern `[a-z0-9]+`).")
A("2. Build a hashed character 2–3-gram TF-IDF index (`2^20` features, sublinear TF, L2).")
A("3. Route each query: if the project-name token exists in the word index (`df > 0`), use word "
  "BM25; otherwise use the character index. This is a proxy for “rare entity is present”; in "
  "production it would be an NER/query-analysis step.")
A("4. Return top-50 and score exact gold-chunk recall.")
A("")
A("## Why this works")
A("")
A("The facts contain unique random project names (`Groth`, `Pathkiath`, ...). Dense bge-small "
  "does not represent those rare names sharply, so it ranks many same-noun facts together. BM25 "
  "matches the rare name exactly and collapses the candidate set. The character index handles the "
  "case where the name is misspelled and the exact token is absent.")
A("")
A("## Caveats")
A("")
A("- **Synthetic, template queries.** All 200 queries contain the exact project name and noun. "
  "Real questions may say “it”, “that project”, or omit the name; those are ambiguous without "
  "conversation state.")
A("- **One typo model.** A last-two-character swap; other typos, transliterations, and aliases may "
  "behave differently.")
A("- **No real-data evaluation.** The corpus is real Discord chatter, but the facts and questions "
  "are synthetic. The next test should use real questions with entity mentions.")
A("- **Character index is memory-heavy.** 37.23 B/token is most of the budget; it can be quantized "
  "or replaced with a smaller n-gram/fuzzy index.")
A("- **Router is a proxy.** It assumes the entity name can be extracted from the query; a real "
  "system needs NER or query rewriting.")
A("")
A("## Next steps")
A("")
A("1. Add dense bge-small as a fallback when no rare name is present (still under 64 B/token).")
A("2. Add a cross-encoder reranker over the top-50; query-time only, no storage cost.")
A("3. Test real user questions and multi-turn references; measure how often the entity name is "
  "present.")
A("4. If real queries often omit names, add a conversation-state resolver that expands “it/that "
  "project” to a candidate entity before retrieval.")
A("")
A("## Files")
A("")
A("- `hybrid_results.json` — router/word/char/RRF recall.")
A("- `bm25_results.json` — word BM25 exact/typo.")
A("- `char_ngram_results.json` — char index exact/typo.")
A("- `hybrid_quality.png` — recall@10 plot.")
A("- `hybrid_report.md` — this report.")

open(os.path.join(WORK, "hybrid_report.md"), "w", encoding="utf-8").write("\n".join(lines))
print("wrote", os.path.join(WORK, "hybrid_report.md"), "lines", len(lines))
