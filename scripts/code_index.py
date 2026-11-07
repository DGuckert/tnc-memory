"""Code-aware memory: symbols, files, error signatures, dependencies, and commands per repo.

Record kinds
  function / method / class   name, qualname, signature, docstring, calls, raised/logged error
                              strings, first lines of the body
  file                        path, imports, module docstring
  error                       an error/log message string, linked to the symbol that emits it
  dependency                  requirements.txt / package.json / build.gradle entries
  command                     shell commands from READMEs, package.json scripts, .sh/.bat files

Python is parsed with ast; JS/TS/Java/Kotlin with brace-matched regex (no tree-sitter dependency).

Search = BM25 over identifier-split tokens (getUserById -> get user by id) + bge-small embeddings
of a natural-language-ish rendering, fused with RRF, optional cross-encoder rerank, and scoped by
repo / kind / path prefix. Repo scope is the code analogue of project scoping for facts.
"""
import ast, json, os, re, time
from collections import defaultdict
import numpy as np

SKIP_DIRS = {"node_modules", ".git", "venv", ".venv", "build", "dist", "third_party", "vendor", "__pycache__",
             "site-packages", ".gradle", ".idea", "out", "target", ".next", "coverage", "migrations"}
EXT = {".py": "python", ".js": "js", ".jsx": "js", ".ts": "js", ".tsx": "js", ".java": "java", ".kt": "kotlin"}
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def split_ident(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return re.sub(r"[_\-./:]+", " ", s).lower()


def toks(s):
    return re.findall(r"[a-z0-9]+", split_ident(s))


# ---------------- parsers ----------------
ERR_CALL = re.compile(r"(?:raise\s+\w+|throw\s+new\s+\w+|log(?:ger)?\.(?:error|warn(?:ing)?|exception)|console\.(?:error|warn)|Log\.[ew]|print)\s*\(\s*f?[\"'`]([^\"'`]{8,160})", re.I)


def parse_python(src, path):
    out = []
    try:
        tree = ast.parse(src)
    except Exception:
        return out, [], ""
    lines = src.splitlines()
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    def visit(node, prefix=""):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(ch, ast.ClassDef) else ("method" if prefix else "function")
                q = f"{prefix}{ch.name}"
                body = "\n".join(lines[ch.lineno - 1: getattr(ch, "end_lineno", ch.lineno)])
                if kind == "class":
                    sig = f"class {ch.name}"
                else:
                    try:
                        sig = f"def {ch.name}({ast.unparse(ch.args)})"
                    except Exception:
                        sig = f"def {ch.name}(...)"
                calls = sorted({c.func.id if isinstance(c.func, ast.Name) else c.func.attr
                                for c in ast.walk(ch) if isinstance(c, ast.Call)
                                and isinstance(c.func, (ast.Name, ast.Attribute))})[:30]
                out.append({"kind": kind, "name": ch.name, "qualname": q, "signature": sig,
                            "doc": (ast.get_docstring(ch) or "")[:400], "line": ch.lineno,
                            "calls": calls, "errors": ERR_CALL.findall(body)[:6], "body": body[:1500]})
                if kind == "class":
                    visit(ch, q + ".")
    visit(tree)
    return out, imports, (ast.get_docstring(tree) or "")[:300]


DEF_PATTERNS = {
    "js": [r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*\(([^)]*)\)",
           r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\(([^)]*)\)|(\w+))\s*=>",
           r"(?:export\s+)?(?:default\s+)?class\s+(\w+)",
           r"^\s+(?:async\s+)?(?!if|for|while|switch|catch|return)(\w+)\s*\(([^)]*)\)\s*\{"],
    "java": [r"(?:public|private|protected|static|final|synchronized|\s)+[\w<>\[\],\s]+\s+(\w+)\s*\(([^)]*)\)\s*(?:throws[\w\s,]+)?\{",
             r"(?:public\s+|abstract\s+|final\s+)*class\s+(\w+)"],
    "kotlin": [r"(?:suspend\s+|private\s+|override\s+|internal\s+|public\s+)*fun\s+(?:<[^>]+>\s*)?(?:\w+\.)?(\w+)\s*\(([^)]*)\)",
               r"(?:data\s+|sealed\s+|abstract\s+|open\s+)*class\s+(\w+)", r"object\s+(\w+)"],
}
IMPORT_RE = {"js": r"(?:import\s.*?from\s+['\"]([^'\"]+)|require\(['\"]([^'\"]+))",
             "java": r"^import\s+([\w.]+)", "kotlin": r"^import\s+([\w.]+)"}


def brace_body(src, start, limit=4000):
    i = src.find("{", start)
    if i < 0 or i - start > 300:
        return src[start:start + 400]
    depth = 0
    for j in range(i, min(len(src), i + limit)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
    return src[start:start + limit]


def parse_braced(src, lang):
    out, seen = [], set()
    for pat in DEF_PATTERNS[lang]:
        for m in re.finditer(pat, src, re.M):
            name = m.group(1)
            if not name or name in ("if", "for", "while", "switch", "catch", "constructor", "return", "super") or (name, m.start()) in seen:
                continue
            seen.add((name, m.start()))
            kind = "class" if "class" in m.group(0) or "object " in m.group(0) else "function"
            body = brace_body(src, m.start())
            pre = src[max(0, m.start() - 400):m.start()]
            doc = ""
            dm = re.search(r"/\*\*(.*?)\*/\s*$", pre, re.S) or re.search(r"((?:^\s*//.*\n)+)\s*$", pre, re.M)
            if dm:
                doc = re.sub(r"^\s*(\*|//)\s?", "", dm.group(1), flags=re.M).strip()[:400]
            calls = sorted(set(re.findall(r"\b([a-zA-Z_]\w{2,})\s*\(", body)) - {name, "if", "for", "while", "switch", "catch", "return"})[:30]
            out.append({"kind": kind, "name": name, "qualname": name, "signature": m.group(0).strip()[:200],
                        "doc": doc, "line": src.count("\n", 0, m.start()) + 1, "calls": calls,
                        "errors": ERR_CALL.findall(body)[:6], "body": body[:1500]})
    imports = [a or b for a, b in re.findall(IMPORT_RE[lang], src, re.M)] if lang == "js" else re.findall(IMPORT_RE[lang], src, re.M)
    return out, imports, ""


def repo_extras(repo, root):
    recs = []
    rp = os.path.join(root, "requirements.txt")
    if os.path.exists(rp):
        for l in open(rp, errors="ignore"):
            l = l.strip()
            if l and not l.startswith("#"):
                recs.append({"kind": "dependency", "name": re.split(r"[<>=~!\[ ;]", l)[0], "text": f"python dependency {l}", "path": "requirements.txt"})
    pj = os.path.join(root, "package.json")
    if os.path.exists(pj):
        try:
            d = json.load(open(pj, errors="ignore"))
            for sec in ("dependencies", "devDependencies"):
                for k, v in d.get(sec, {}).items():
                    recs.append({"kind": "dependency", "name": k, "text": f"npm {sec} {k} {v}", "path": "package.json"})
            for k, v in d.get("scripts", {}).items():
                recs.append({"kind": "command", "name": f"npm run {k}", "text": f"npm run {k}: {v}", "path": "package.json"})
        except Exception:
            pass
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        rel = os.path.relpath(dirpath, root)
        if rel.count(os.sep) > 2:
            continue
        for f in files:
            p = os.path.join(dirpath, f)
            relp = os.path.normpath(os.path.join(rel, f))
            if f.endswith((".gradle", ".gradle.kts")):
                for m in re.findall(r"(?:implementation|api|kapt|ksp)\s*\(?\s*[\"']([^\"']+)", open(p, errors="ignore").read()):
                    recs.append({"kind": "dependency", "name": m.split(":")[1] if ":" in m else m, "text": f"gradle dependency {m}", "path": relp})
            elif f.lower().startswith("readme") and f.lower().endswith(".md"):
                txt = open(p, errors="ignore").read()
                for block in re.findall(r"```(?:bash|sh|shell|powershell|ps1|cmd|bat)?\n(.*?)```", txt, re.S):
                    for line in block.splitlines():
                        line = line.strip()
                        if 4 < len(line) < 200 and not line.startswith("#"):
                            recs.append({"kind": "command", "name": line[:60], "text": f"command from {relp}: {line}", "path": relp})
            elif f.endswith((".sh", ".bat", ".ps1")) and os.path.getsize(p) < 20000:
                body = open(p, errors="ignore").read()
                head = " ".join(l.strip() for l in body.splitlines() if l.strip() and not l.strip().lower().startswith(("@echo", "#!/")))[:300]
                recs.append({"kind": "command", "name": f, "text": f"script {relp}: {head}", "path": relp})
    return recs


def render(r):
    """Text used for embedding and BM25."""
    if r["kind"] in ("dependency", "command", "error"):
        return f"{r['repo']} {r['path']} {r['text']}"
    if r["kind"] == "file":
        return f"{r['repo']} file {r['path']} {split_ident(r['path'])}. imports {' '.join(r.get('imports', [])[:20])}. {r.get('doc', '')}"
    parts = [r["repo"], r["path"], f"{r['kind']} {r['qualname']} ({split_ident(r['qualname'])})", r["signature"], r["doc"],
             "errors: " + "; ".join(r["errors"]) if r["errors"] else "",
             "calls " + " ".join(split_ident(c) for c in r["calls"][:15]), r["body"][:600]]
    return "\n".join(p for p in parts if p)


class CodeIndex:
    def __init__(self, repos, cache=None, device="cuda", rerank=True):
        """repos: {name: root_path}."""
        from sentence_transformers import SentenceTransformer
        self.repos = dict(repos)
        self.records = []
        t0 = time.time()
        for name, root in self.repos.items():
            for dirpath, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
                for f in files:
                    ext = os.path.splitext(f)[1]
                    if ext not in EXT:
                        continue
                    p = os.path.join(dirpath, f)
                    if os.path.getsize(p) > 300_000:
                        continue
                    rel = os.path.relpath(p, root).replace(os.sep, "/")
                    src = open(p, errors="ignore").read()
                    lang = EXT[ext]
                    syms, imports, mdoc = parse_python(src, rel) if lang == "python" else parse_braced(src, lang)
                    self.records.append({"repo": name, "path": rel, "kind": "file", "name": os.path.basename(rel),
                                         "imports": imports[:40], "doc": mdoc, "lang": lang})
                    for s in syms:
                        s.update(repo=name, path=rel, lang=lang)
                        self.records.append(s)
                        for e in s["errors"]:
                            self.records.append({"repo": name, "path": rel, "kind": "error", "name": e[:60],
                                                 "text": f"error message '{e}' raised in {s['qualname']}", "owner": s["qualname"]})
            for r in repo_extras(name, root):
                r["repo"] = name
                self.records.append(r)
        self.build_s = time.time() - t0
        self.texts = [render(r) for r in self.records]
        self.repo_arr = np.array([r["repo"] for r in self.records])
        self.kind_arr = np.array([r["kind"] for r in self.records])
        # BM25 over identifier-split tokens
        self.doc_toks = [toks(t) for t in self.texts]
        self.df = defaultdict(int)
        for t in self.doc_toks:
            for w in set(t):
                self.df[w] += 1
        self.dl = np.array([len(t) for t in self.doc_toks], dtype=np.float32)
        self.post = defaultdict(list)
        for i, t in enumerate(self.doc_toks):
            c = defaultdict(int)
            for w in t:
                c[w] += 1
            for w, n in c.items():
                self.post[w].append((i, n))
        self.emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
        self.emb.max_seq_length = 384
        self.vec = self.emb.encode(self.texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False).astype(np.float32)
        self.ce = None
        if rerank:
            from sentence_transformers import CrossEncoder
            self.ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_length=384)

    def entity_names(self):
        return list(self.repos)

    def stats(self):
        from collections import Counter
        return {"records": len(self.records), "by_kind": dict(Counter(self.kind_arr.tolist())),
                "by_repo": dict(Counter(self.repo_arr.tolist())), "parse_s": round(self.build_s, 1)}

    def bm25(self, q, k1=1.2, b=0.75):
        n, avg = len(self.records), float(self.dl.mean())
        s = np.zeros(n, dtype=np.float32)
        for w in set(toks(q)):
            if w not in self.post or w in STOP:
                continue
            idf = np.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
            for i, c in self.post[w]:
                s[i] += idf * c * (k1 + 1) / (c + k1 * (1 - b + b * self.dl[i] / avg))
        return s

    def search(self, query, k=10, repo=None, kinds=None, mode="hybrid", rerank=True, pool=50, rrf_k=60):
        mask = np.ones(len(self.records), dtype=bool)
        if repo:
            mask &= self.repo_arr == repo
        if kinds:
            mask &= np.isin(self.kind_arr, list(kinds))
        if not mask.any():
            return []
        lists = []
        if mode in ("bm25", "hybrid"):
            s = self.bm25(query); s[~mask] = -np.inf
            lists.append(np.argsort(-s)[:pool * 2])
        if mode in ("dense", "hybrid"):
            q = self.emb.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0]
            s = self.vec @ q; s[~mask] = -np.inf
            lists.append(np.argsort(-s)[:pool * 2])
        fused = defaultdict(float)
        for ids in lists:
            for r, i in enumerate(ids):
                if mask[i]:
                    fused[int(i)] += 1 / (rrf_k + r + 1)
        cand = sorted(fused.items(), key=lambda x: -x[1])[:pool]
        if rerank and self.ce is not None and cand:
            ids = [i for i, _ in cand]
            sc = self.ce.predict([(query, self.texts[i][:1200]) for i in ids], batch_size=32, show_progress_bar=False)
            cand = sorted(zip(ids, map(float, sc)), key=lambda x: -x[1])
        out = []
        for i, s in cand[:k]:
            r = self.records[i]
            label = r.get("qualname") or r.get("name")
            summ = f"{r['kind']} {label}" + (f" - {r['doc'][:120]}" if r.get("doc") else "") + (f" [{r['signature'][:100]}]" if r.get("signature") else "")
            if r["kind"] in ("dependency", "command", "error"):
                summ = r["text"][:200]
            out.append({"idx": i, "score": s, "summary": summ,
                        "ref": {"repo": r["repo"], "path": r["path"], "kind": r["kind"], "name": label, "line": r.get("line")}})
        return out


STOP = set("the a an and or of to in on for is it that this what was who how did does do with where which "
           "we our my i you me be are when there function code thing stuff".split())
