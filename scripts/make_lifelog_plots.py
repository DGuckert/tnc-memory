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
         "cards": "cards v1 (raw extraction)", "cards_v2": "cards v2 (hygiene + indexes)",
         "cards_v3": "cards v3 (+ batch hygiene + recall pass)",
         "cards_v2_oracle": "cards + indexes, perfect extraction", "agent": "agent (tools)"}
COLOR = {"full": "#555555", "rag": "#c2571a", "cards": "#c6d8ef", "cards_v2": "#8fb3e0", "cards_v3": "#1f5aa6",
         "cards_v2_oracle": "#0b2f63", "agent": "#2a7f62"}
ORDER = ["full", "rag", "cards", "cards_v2", "cards_v3", "cards_v2_oracle", "agent"]
XT = ["full\ncontext", "RAG", "cards\nv1", "cards\nv2", "cards\nv3", "perfect\nextraction", "agent"]


def load(name):
    """Merge the v1 run and the v2 run of a log. v2's agent (grammar-bounded) replaces v1's, which never
    called its tools; v1's cards_oracle (no indexes) is superseded by cards_v2_oracle."""
    d = None
    for suffix in ("", "_v2", "_v3"):
        p = os.path.join(W, f"lifelog_eval_{name}{suffix}.json")
        if os.path.exists(p):
            x = json.load(open(p))
            if suffix == "_v3" and "cards_v2" in x["conditions"]:  # v3 runs the cards_v2 condition on the v3 store
                x["conditions"]["cards_v3"] = x["conditions"].pop("cards_v2")
            if d is None:
                d = x
            else:
                d["conditions"].update(x["conditions"])
    return d


def by_type(d, title, fname):
    conds = [c for c in ORDER if c in d["conditions"]]
    fig, ax = plt.subplots(figsize=(12, 5.4))
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
    ax.set_title(title); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=130); plt.close(fig)


def overall(small, big):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for d, name, off in ((small, "small log (~7K tokens)", -0.2), (big, "big log (~600K tokens)", 0.2)):
        if not d:
            continue
        conds = [c for c in ORDER if c in d["conditions"]]
        for c in conds:
            i = ORDER.index(c)
            acc = d["conditions"][c]["summary"]["accuracy"]
            ax[0].bar(i + off, acc, 0.38, color=COLOR[c], alpha=1.0 if off < 0 else 0.55,
                      hatch=None if off < 0 else "//", edgecolor="white")
            ax[0].text(i + off, acc + 0.015, f"{acc:.0%}", ha="center", fontsize=7)
            tok = d["conditions"][c]["summary"]["prompt_tokens_median"]
            ax[1].scatter(tok, acc, color=COLOR[c], s=70, marker="o" if off < 0 else "s",
                          label=f"{LABEL[c].split(' (')[0]}: {name.split(' (')[0]}")
        if "full" not in d["conditions"]:
            ax[0].text(ORDER.index("full") + off, 0.02, "can't\nfit", ha="center", fontsize=7, color="#555")
    ax[0].set_xticks(range(len(ORDER))); ax[0].set_xticklabels(XT, fontsize=8)
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("accuracy, all question types")
    ax[0].set_title("solid = small log, hatched = 600K-token log"); ax[0].grid(axis="y", alpha=.3)
    ax[1].set_xscale("log"); ax[1].set_xlabel("prompt tokens per question (median; agent = summed over steps)")
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("accuracy"); ax[1].grid(alpha=.3)
    ax[1].set_title("accuracy vs tokens spent (circle = small log, square = 600K log)"); ax[1].legend(fontsize=6, loc="lower right", ncol=2)
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
