import json, re, math, time
from collections import defaultdict
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

DEFAULT_MIDDLEMAN = "/home/daniel/tnc/training_runs/tool-135m-lora"
DEFAULT_READER = "/home/daniel/tnc/training_runs/middleman-lora"
DEFAULT_RERANKER = "/home/daniel/tnc/training_runs/reranker-mini"
NF = 2 ** 20

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def extract_note(text):
    m = re.findall(r"Note to self:.*", text)
    return m[-1].strip() if m else text.strip()

def canonical_from_note(note):
    m = re.search(r"the (.+?) for project ([A-Za-z0-9]+) is", note)
    if not m:
        return None
    return f"what is the {m.group(1)} for project {m.group(2)}?"

def load_adapter(path):
    cfg = PeftConfig.from_pretrained(path)
    tok = AutoTokenizer.from_pretrained(path)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype=torch.float16)
    base.to("cuda")
    model = PeftModel.from_pretrained(base, path).eval()
    model.config.use_cache = True
    return tok, model

def osa(a, b):
    prev2 = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
            if prev2 is not None and i > 1 and j > 1 and ca == b[j-2] and a[i-2] == cb:
                cur[j] = min(cur[j], prev2[j-2] + 1)
        prev2, prev = prev, cur
    return prev[-1]

class MemoryGateway:
    def __init__(self, docs, middleman=DEFAULT_MIDDLEMAN, reader=DEFAULT_READER,
                 reranker=DEFAULT_RERANKER, top_k=5, max_new_tokens=48):
        self.docs = docs
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self._build_word()
        self._build_char()
        self.entity_map = {x.lower(): x for d in docs for x in re.findall(r"project ([A-Za-z0-9]+) is", d)}
        self.entities = sorted(self.entity_map)
        self.mtok, self.llm = load_adapter(middleman)
        if reader and reader != middleman:
            self.reader_tok, self.reader_llm = load_adapter(reader)
        else:
            self.reader_tok, self.reader_llm = self.mtok, self.llm
        self.rtok = AutoTokenizer.from_pretrained(reranker)
        self.rmodel = AutoModelForSequenceClassification.from_pretrained(reranker, num_labels=1).to("cuda").eval()

    def _build_word(self):
        self.doc_len = np.array([len(tokenize(d)) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean())
        self.postings = defaultdict(list)
        for i, d in enumerate(self.docs):
            tf = defaultdict(int)
            for t in tokenize(d):
                tf[t] += 1
            for t, c in tf.items():
                self.postings[t].append((i, c))
        self.idf = {t: math.log(1.0 + (len(self.docs) - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}
        self.k1, self.b = 1.2, 0.75

    def _build_char(self):
        self.vec = HashingVectorizer(n_features=NF, analyzer="char", ngram_range=(2, 3),
                                     lowercase=True, alternate_sign=False, norm=None, dtype=np.float32)
        X = self.vec.transform(self.docs)
        self.tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
        self.X = self.tfidf.fit_transform(X)

    def _topk(self, scores, k):
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        return idx[np.argsort(-scores[idx])]

    def _correct_query(self, query):
        out = []
        changed = False
        for t in tokenize(query):
            if t in self.postings or len(t) < 4:
                out.append(t)
                continue
            cands = sorted((osa(t, e), e) for e in self.entities)[:3]
            if cands and cands[0][0] <= 2:
                out.append(cands[0][1])
                changed = True
            else:
                out.append(t)
        return " ".join(out), changed

    def search(self, query, project=None, mode=None, k=50):
        if mode is None:
            corrected, changed = self._correct_query(query)
            if changed:
                query = corrected
                mode = "word"
            else:
                mode = "word" if project and project.lower() in self.postings else "char"
        if mode == "word":
            s = np.zeros(len(self.docs), dtype=np.float32)
            for t in set(tokenize(query)):
                p = self.postings.get(t)
                if not p:
                    continue
                for i, c in p:
                    denom = c + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    s[i] += self.idf[t] * c * (self.k1 + 1.0) / denom
            ids = self._topk(s, k)
        else:
            qv = self.tfidf.transform(self.vec.transform([query]))
            s = (self.X @ qv.T).toarray().ravel()
            ids = self._topk(s, k)
        return ids, mode

    def rerank(self, query, ids, k=5):
        passages = [self.docs[int(i)] for i in ids]
        enc = self.rtok([query] * len(passages), passages, truncation=True, max_length=256,
                        padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            scores = self.rmodel(**enc).logits.view(-1).float().cpu().numpy()
        order = np.argsort(-scores)[:k]
        return [int(ids[i]) for i in order]

    def tool_call(self, messages):
        prompt = self.mtok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.mtok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.llm.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                    pad_token_id=self.mtok.eos_token_id,
                                    eos_token_id=self.mtok.eos_token_id)
        text = self.mtok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            return json.loads(text), text
        except Exception:
            return None, text

    def answer(self, question, ids):
        notes = [extract_note(self.docs[int(i)]) for i in ids]
        user = "Memory notes:\n" + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
        user += f"\nQuestion: {question}"
        msgs = [{"role": "system", "content": "You answer only from the memory notes. If the answer is not present, say exactly: I don't know."},
                {"role": "user", "content": user}]
        prompt = self.reader_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.reader_tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.reader_llm.generate(**enc, max_new_tokens=12, do_sample=False,
                                           pad_token_id=self.reader_tok.eos_token_id,
                                           eos_token_id=self.reader_tok.eos_token_id)
        return self.reader_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def find_explicit_project(self, messages):
        text = messages[-1].get("content", "") if messages else ""
        toks = tokenize(text)
        for t in toks:
            if t in self.entity_map:
                return self.entity_map[t]
        for t in toks:
            if t in self.postings or len(t) < 4:
                continue
            cands = sorted((osa(t, e), e) for e in self.entities)[:3]
            if cands and cands[0][0] <= 2:
                return self.entity_map[cands[0][1]]
        return None

    def find_context_project(self, messages):
        text = messages[-1].get("content", "") if messages else ""
        if not re.search(r"\b(it|its|that project|the project)\b", text, re.I):
            return None
        found = {}
        for m in messages[:-1]:
            for t in tokenize(m.get("content", "")):
                if t in self.entity_map:
                    found[self.entity_map[t]] = True
        if len(found) == 1:
            return next(iter(found))
        return None

    def handle(self, messages):
        t0 = time.perf_counter()
        project = self.find_explicit_project(messages) or self.find_context_project(messages)
        if project:
            query = messages[-1].get("content", "")
            if project.lower() not in tokenize(query):
                query = f"{query} project {project}"
            tool = {"tool": "search_memory",
                    "args": {"project": project, "query": query}}
            raw = "project shortcut"
        else:
            tool, raw = self.tool_call(messages)
        t1 = time.perf_counter()
        if not isinstance(tool, dict) or tool.get("tool") != "search_memory":
            return {"tool": tool, "raw": raw, "answer": None, "notes": [],
                    "timings": {"tool": t1 - t0, "total": t1 - t0}}
        args = tool.get("args", {})
        query = args.get("query", "")
        project = args.get("project", "")
        ids, mode = self.search(query, project)
        t2 = time.perf_counter()
        canon0 = canonical_from_note(extract_note(self.docs[int(ids[0])])) or query
        ids = self.rerank(canon0, ids, self.top_k)
        t3 = time.perf_counter()
        notes = [extract_note(self.docs[int(i)]) for i in ids]
        canon = canonical_from_note(notes[0]) or canon0
        answer = self.answer(canon, ids)
        t4 = time.perf_counter()
        return {"tool": tool, "raw": raw, "mode": mode, "query": canon,
                "answer": answer, "notes": notes,
                "timings": {"tool": t1 - t0, "search": t2 - t1,
                            "rerank": t3 - t2, "reader": t4 - t3,
                            "total": t4 - t0}}
