"""Plots for the v2 report: top_k sweep (360M reader vs host model) and retrieval comparisons."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/daniel/tnc/work"
OUT = "/home/daniel/tnc-repo/results/training"


def load(p):
    p = os.path.join(W, p)
    return json.load(open(p)) if os.path.exists(p) else None


def sweep_plot():
    old = load("topk_sweep.json")
    nocanon = load("topk_sweep_nocanon.json")
    host = load("host_reader_gemma3-1b.json")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    def line(d, key, label, style, color):
        if not d:
            return
        ks = sorted(int(k) for k in d)
        ax[0].plot(ks, [d[str(k)][key] for k in ks], style, color=color, marker="o", label=label)
    s = old["bench975"]["sweep"]
    line(s, "accuracy", "360M reader, ranked (v1 gateway)", "-", "#1f5aa6")
    line(s, "accuracy_shuffled", "360M reader, shuffled notes", "--", "#1f5aa6")
    if nocanon:
        line(nocanon["bench975"]["sweep"], "accuracy", "360M reader, raw query (no canon)", ":", "#1f5aa6")
    if host:
        line(host["bench975"]["sweep"], "accuracy", "gemma-3-1b host, ranked", "-", "#c2571a")
        line(host["bench975"]["sweep"], "accuracy_shuffled", "gemma-3-1b host, shuffled", "--", "#c2571a")
    qwen = load("host_reader_qwen3-30b-b975.json")
    if qwen:
        line(qwen["bench975"]["sweep"], "accuracy", "Qwen3-30B-A3B host, ranked (n=100)", "-", "#2a7f62")
        line(qwen["bench975"]["sweep"], "accuracy_shuffled", "Qwen3-30B-A3B host, shuffled", "--", "#2a7f62")
    ax[0].set_xscale("log"); ax[0].set_xticks([1, 5, 10, 20, 50]); ax[0].set_xticklabels(["1", "5", "10", "20", "50"])
    ax[0].set_ylim(0, 1.02); ax[0].set_xlabel("memory notes given to the reader (k)"); ax[0].set_ylabel("answer accuracy")
    ax[0].set_title("9.75M store, 200 conversations, structured shortcut off"); ax[0].legend(fontsize=8, loc="lower left")
    ax[0].grid(alpha=.3)
    ks = sorted(int(k) for k in s)
    ax[1].plot(ks, [s[str(k)]["reader_prompt_tokens"]["median"] for k in ks], "-o", color="#1f5aa6", label="prompt tokens (median)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks([1, 5, 10, 20, 50]); ax[1].set_xticklabels(["1", "5", "10", "20", "50"])
    ax[1].set_xlabel("k"); ax[1].set_ylabel("tokens"); ax[1].grid(alpha=.3)
    a2 = ax[1].twinx()
    a2.plot(ks, [s[str(k)]["latency_ms"]["total"]["median"] for k in ks], "-s", color="#555", label="360M total latency (ms, median)")
    a2.set_ylabel("ms")
    ax[1].set_title("cost of more notes")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "topk_sweep.png"), dpi=130)


def retrieval_plot():
    real = load("real_eval/real_eval_metrics.json")
    code = load("code_eval/code_eval_metrics.json")
    if not real:
        return
    fig, ax = plt.subplots(1, 2 if code else 1, figsize=(12 if code else 6, 4.4))
    ax = ax if code else [ax]
    names = ["bm25", "dense_msg", "dense_ctx", "hybrid", "hybrid+rerank"]
    for j, (scope, col) in enumerate((("global", "#888"), ("channel_oracle", "#2a7f62"))):
        vals = [real["conditions"][f"{n}[msg]|{scope}" if n == "hybrid+rerank" else f"{n}|{scope}"]["exact"]["r@10"] for n in names]
        ax[0].bar([i + j * 0.38 for i in range(len(names))], vals, 0.38, color=col, label=f"{scope} r@10")
    ax[0].set_xticks([i + 0.19 for i in range(len(names))]); ax[0].set_xticklabels(names, rotation=20, fontsize=8)
    ax[0].set_ylim(0, 1); ax[0].set_title(f"real Discord recall, {real['n']} questions / {real['n_messages']:,} msgs")
    ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)
    if code:
        names = ["bm25", "dense", "hybrid", "hybrid+rerank"]
        for j, (scope, col) in enumerate((("global", "#888"), ("repo_scoped", "#2a7f62"))):
            vals = [code["conditions"][f"{n}|{scope}"]["r@1"] for n in names]
            ax[1].bar([i + j * 0.38 for i in range(len(names))], vals, 0.38, color=col, label=f"{scope} r@1")
        ax[1].set_xticks([i + 0.19 for i in range(len(names))]); ax[1].set_xticklabels(names, rotation=20, fontsize=8)
        ax[1].set_ylim(0, 1); ax[1].set_title(f"vague coding queries, {code['n']} q / {code['index']['records']:,} records")
        ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "retrieval_v2.png"), dpi=130)


if __name__ == "__main__":
    sweep_plot(); retrieval_plot(); print("plots written")
