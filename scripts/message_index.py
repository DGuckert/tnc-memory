"""Semantic + lexical message index with channel / author / time scoping.

Memory units are single messages, but each one is also embedded with its conversational
context (the 2 previous messages in the same channel), because most chat messages are
meaningless alone ("every update i gotta fix it").

Retrievers
  dense_msg   bge-small over "author: text"
  dense_ctx   bge-small over the 3-message window ending at the message
  bm25        sparse BM25 over message text (+ author)
Fusion: reciprocal-rank fusion of the three, then optional cross-encoder rerank of the top N
using the context window as the passage.

Scoping: every search takes channel=, author=, and time=(lo, hi) on the message position
(the corpus has no timestamps; position within the export is the time proxy). Scopes are
applied as masks BEFORE top-k, so a scoped search never loses the target to out-of-scope hits.

Build cost (RTX 2060): ~2 x 603K embeddings, cached to WORK/msgidx/*.npy.
"""
import json, math, os, re, time
from collections import defaultdict
import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

WORK = "/home/daniel/tnc/work/msgidx"
CORPUS = "/home/daniel/tnc/corpus.jsonl"
EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_corpus(path=CORPUS):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def context_windows(msgs, n_prev=2, max_chars=600):
    last = defaultdict(list)
    out = []
    for m in msgs:
        h = last[m["ch"]]
        line = f"{m['a']}: {m['t']}"
        win = "\n".join([x[:200] for x in h[-n_prev:]] + [line])
        out.append(win[-max_chars:])
        h.append(line)
        if len(h) > n_prev:
            del h[0]
    return out


def embed_texts(texts, model, bs=256):
    vecs = model.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)
    return vecs.astype(np.float16)


class MessageIndex:
    def __init__(self, msgs=None, device="cuda", rerank=True, build_missing=True):
        from sentence_transformers import SentenceTransformer
        os.makedirs(WORK, exist_ok=True)
        self.msgs = msgs if msgs is not None else load_corpus()
        self.device = device
        n = len(self.msgs)
        self.texts = [m["t"] for m in self.msgs]
        self.windows = context_windows(self.msgs)
        self.channels = np.array([m["ch"] for m in self.msgs])
        self.authors = np.array([m["a"] for m in self.msgs])
        self.author_lc = {a.lower(): a for a in set(self.authors.tolist())}
        self.emb = SentenceTransformer(EMB_MODEL, device=device)
        self.emb.max_seq_length = 256
        self.vec = {}
        for name, texts in (("msg", [f"{m['a']}: {m['t']}" for m in self.msgs]), ("ctx", self.windows)):
            p = os.path.join(WORK, f"emb_{name}_{n}.npy")
            if not os.path.exists(p):
                if not build_missing:
                    raise FileNotFoundError(p)
                t0 = time.time()
                order = np.argsort([len(t) for t in texts])  # length-sorted batches = less padding
                v = np.zeros((n, 384), dtype=np.float16)
                chunk = 50000
                for s in range(0, n, chunk):
                    ids = order[s:s + chunk]
                    v[ids] = embed_texts([texts[i] for i in ids], self.emb)
                    print(f"  embed {name} {min(s + chunk, n)}/{n} {time.time() - t0:.0f}s", flush=True)
                np.save(p, v)
            self.vec[name] = torch.from_numpy(np.load(p)).to(device)
        self._build_bm25()
        self.rerank_model = None
        if rerank:
            from sentence_transformers import CrossEncoder
            self.rerank_model = CrossEncoder(RERANKER, device=device, max_length=320)

    # ---------- lexical ----------
    def _build_bm25(self, k1=1.2, b=0.75):
        self.cv = CountVectorizer(token_pattern=r"[A-Za-z0-9][A-Za-z0-9.\-_]*[A-Za-z0-9]|[A-Za-z0-9]",
                                  lowercase=True, dtype=np.float32, min_df=1)
        docs = [f"{m['a']} {m['t']}" for m in self.msgs]
        tf = self.cv.fit_transform(docs).tocsr()
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        n = tf.shape[0]
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
        data = tf.data * (k1 + 1) / denom
        self.bm25 = sparse.csc_matrix((data.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.vocab = self.cv.vocabulary_
        self.stop = set("the a an and or of to in on for is it that this what was who how did does do with "
                        "about someone say said saying be are i you they he she we my your their at as by from "
                        "but not so if just like any there which when where why can could would should me".split())

    def bm25_scores(self, query):
        toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-_]*[A-Za-z0-9]|[A-Za-z0-9]", query.lower())
                if t not in self.stop]
        cols = [self.vocab[t] for t in set(toks) if t in self.vocab]
        s = np.zeros(self.bm25.shape[0], dtype=np.float32)
        if cols:
            sub = self.bm25[:, cols] @ self.idf[cols]
            s = np.asarray(sub).ravel().astype(np.float32)
        return s

    # ---------- scoping ----------
    def scope_mask(self, channel=None, author=None, time=None):
        m = np.ones(len(self.msgs), dtype=bool)
        if channel:
            m &= self.channels == channel
        if author:
            a = self.author_lc.get(author.lower(), author)
            m &= self.authors == a
        if time:
            lo, hi = time
            idx = np.arange(len(self.msgs))
            m &= (idx >= (lo or 0)) & (idx < (hi or len(self.msgs)))
        return m

    # ---------- search ----------
    def dense_scores(self, query, which):
        q = self.emb.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_tensor=True)
        q = q.to(self.device).half()
        return (self.vec[which] @ q[0]).float()

    @staticmethod
    def _rank(scores, mask, k):
        s = scores.copy()
        s[~mask] = -np.inf
        k = min(k, int(mask.sum()))
        idx = np.argpartition(-s, k - 1)[:k]
        return idx[np.argsort(-s[idx])]

    def search(self, query, k=10, channel=None, author=None, time=None, mode="hybrid",
               rerank=True, pool=50, rrf_k=60, weights=None, rerank_passage="msg"):
        """Return list of (msg_idx, score). mode: bm25 | dense_msg | dense_ctx | dense | hybrid."""
        mask = self.scope_mask(channel, author, time)
        if mask.sum() == 0:
            return []
        lists = {}
        if mode in ("bm25", "hybrid"):
            lists["bm25"] = self._rank(self.bm25_scores(query), mask, pool * 2)
        if mode in ("dense_msg", "dense", "hybrid"):
            lists["msg"] = self._rank(self.dense_scores(query, "msg").cpu().numpy(), mask, pool * 2)
        if mode in ("dense_ctx", "dense", "hybrid"):
            lists["ctx"] = self._rank(self.dense_scores(query, "ctx").cpu().numpy(), mask, pool * 2)
        weights = weights or {"bm25": 1.0, "msg": 1.0, "ctx": 1.0}
        fused = defaultdict(float)
        for name, ids in lists.items():
            for r, i in enumerate(ids):
                fused[int(i)] += weights.get(name, 1.0) / (rrf_k + r + 1)
        cand = sorted(fused.items(), key=lambda x: -x[1])[:pool]
        if rerank and self.rerank_model is not None and cand:
            ids = [i for i, _ in cand]
            # "window" scored the context window, which let the NEXT message (whose window contains
            # the target) outrank the target itself: exact r@1 0.38 vs same-exchange r@1 0.93.
            # "msg" scores the message alone; "ctx" scores the message with its preceding context marked.
            psg = [self.windows[i] if rerank_passage == "window" else
                   (f"{self.msgs[i]['a']}: {self.msgs[i]['t']}" if rerank_passage == "msg" else
                    f"{self.msgs[i]['a']}: {self.msgs[i]['t']} (context: {self.windows[i][:-len(self.msgs[i]['t'])][-300:]})")
                   for i in ids]
            sc = self.rerank_model.predict([(query, p) for p in psg], batch_size=64,
                                           show_progress_bar=False)
            cand = sorted(zip(ids, map(float, sc)), key=lambda x: -x[1])
        return cand[:k]

    def note(self, i, with_context=True):
        m = self.msgs[i]
        return {"idx": i, "channel": m["ch"], "author": m["a"], "text": m["t"],
                "context": self.windows[i] if with_context else None}
