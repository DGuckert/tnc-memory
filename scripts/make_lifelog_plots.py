"""Plots for the life-log ("memory that behaves like context") eval. Re-runnable: plots whatever
conditions exist in work/lifelog/lifelog_eval_{small,big}.json so far."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/daniel/tnc/work/lifelog"
OUT = "/home/daniel/tnc-repo/results/training"
TYPES = ["current", "past", "history", "aggregate", "count", "multi_hop", "temporal"]
LABEL = {"full": "full context (whole log in prompt)", "rag": "RAG (top-10 messages)",
         "cards": "cards (extracted at write time)", "cards_oracle": "cards (perfect extraction)",
         "agent": "agent (tools: search + fact store)"}
COLOR = {"full": "#555555", "rag": "#c2571a", "cards": "#1f5aa6", "cards_oracle": "#8fb3e0", "agent": "#2a7f62"}
ORDER = ["full", "rag", "cards", "cards_oracle", "agent"]


def load(name):
    p = os.path.join(W, f"lifelog_eval_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def by_type(d, title, fname):
    conds = [c for c in ORDER if c in d["conditions"]]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    w = 0.8 / max(1, len(conds))
    for j, c in enumerate(conds):
        bt = d["conditions"][c]["summary"]["by_type"]
        xs = [i + (j - (len(conds) - 1) / 2) * w for i in range(len(TYPES))]
        ax.bar(xs, [bt.get(t, 0) for t in TYPES], w * 0.92, color=COLOR[c],
               label=f"{LABEL[c]}: {d['conditions'][c]['summary']['accuracy']:.0%} overall")
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels(["current value", "past value", "full history", "which projects…", "how many…",
                        "multi-hop", "which first…"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy"); ax.grid(axis="y", alpha=.3)
    ax.set_title(title); ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=130); plt.close(fig)


def overall(small, big):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for d, name, off in ((small, "small log (~7K tokens)", -0.2), (big, "big log (~600K tokens)", 0.2)):
        if not d:
            continue
        conds = [c for c in ORDER if c in d["conditions"]]
        for c in conds:
            i = ORDER.index(c)
            acc = d["conditions"][c]["summary"]["accuracy"]
            ax[0].bar(i + off, acc, 0.38, color=COLOR[c], alpha=1.0 if off < 0 else 0.55,
                      hatch=None if off < 0 else "//", edgecolor="white")
            ax[0].text(i + off, acc + 0.015, f"{acc:.0%}", ha="center", fontsize=8)
            tok = d["conditions"][c]["summary"]["prompt_tokens_median"]
            ax[1].scatter(tok, acc, color=COLOR[c], s=70, marker="o" if off < 0 else "s",
                          label=f"{c} — {name}")
    ax[0].set_xticks(range(len(ORDER))); ax[0].set_xticklabels(["full\ncontext", "RAG", "cards", "cards\n(perfect)", "agent"])
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("accuracy, all question types")
    ax[0].set_title("solid = small log, hatched = 600K-token log"); ax[0].grid(axis="y", alpha=.3)
    ax[1].set_xscale("log"); ax[1].set_xlabel("prompt tokens per question (median; agent = summed over steps)")
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("accuracy"); ax[1].grid(alpha=.3)
    ax[1].set_title("accuracy vs tokens spent"); ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "lifelog_overall.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    small, big = load("small"), load("big")
    if small:
        by_type(small, f"Life-log questions by type — small log ({small['n_messages']} msgs), "
                       f"{small['n_questions']} questions, Qwen3-30B host", "lifelog_by_type_small.png")
    if big:
        by_type(big, f"Life-log questions by type — big log ({big['n_messages']:,} msgs, ~600K tokens)",
                "lifelog_by_type_big.png")
    overall(small, big)
    print("lifelog plots written")
