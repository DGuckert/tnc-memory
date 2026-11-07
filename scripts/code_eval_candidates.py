"""Build the code index over local repos and sample target symbols for a vague-query eval."""
import json, random, sys, pickle
sys.path.insert(0, "/home/daniel/tnc")
from code_index import CodeIndex

REPOS = {
    "voice-assistant": "/mnt/c/Users/Daniel/voice-assistant",
    "bus-alarm": "/mnt/c/Users/Daniel/bus-alarm",
    "jarvis-launcher": "/mnt/c/Users/Daniel/jarvis-launcher",
    "discord-ai-response": "/mnt/c/Users/Daniel/Discord-ai-response",
    "chessforge": "/mnt/c/Users/Daniel/ChessForge",
    "butler": "/mnt/c/Users/Daniel/ButlerMusic",
    "tnc-memory": "/home/daniel/tnc-repo",
    # third-party clones on the same machine: realistic distractors for unscoped search
    "flowsint": "/mnt/c/Users/Daniel/flowsint",
    "strata": "/mnt/c/Users/Daniel/Strata",
}
OWN = ["voice-assistant", "bus-alarm", "jarvis-launcher", "discord-ai-response", "chessforge", "butler", "tnc-memory"]

if __name__ == "__main__":
    ci = CodeIndex(REPOS, rerank=False)
    print(json.dumps(ci.stats(), indent=1))
    rng = random.Random(77)
    pool = [i for i, r in enumerate(ci.records) if r["repo"] in OWN and r["kind"] in ("function", "method")
            and r["body"].count("\n") >= 6 and not r["name"].startswith("__") and r["name"] not in ("main", "run", "init")]
    per = {}
    for i in pool:
        per.setdefault(ci.records[i]["repo"], []).append(i)
    pick = []
    for repo, ids in per.items():
        pick += rng.sample(ids, min(len(ids), 16))
    rng.shuffle(pick)
    with open("/home/daniel/tnc/work/code_eval/candidates.jsonl", "w") as f:
        for n, i in enumerate(pick):
            r = ci.records[i]
            f.write(json.dumps({"n": n, "idx": i, "repo": r["repo"], "path": r["path"], "qualname": r["qualname"],
                                "doc": r["doc"][:200], "body": r["body"][:700]}) + "\n")
    print("candidates", len(pick), {k: len(v) for k, v in per.items()})
