import json, re, math
from collections import defaultdict
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_LLM = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_MIDDLEMAN = "/home/daniel/tnc/training_runs/middleman-lora"
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

class MemoryGateway:
    def __init__(self, docs, middleman=DEFAULT_MIDDLEMAN, reranker=DEFAULT_RERANKER,
                 top_k=5, max_new_tokens=64):
        self.docs = docs
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self._build_word()
        self._build_char()
        self.mtok = AutoTokenizer.from_pretrained(middleman)
        base = AutoModelForCausalLM.from_pretrained(BASE_LLM, torch_dtype=torch.float16).to("cuda")
        self.llm = PeftModel.from_pretrained(base, middleman).eval()
        self.llm.config.use_cache = True
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

    def search(self, query, project=None, mode=None, k=50):
        if mode is None:
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
        prompt = self.mtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.mtok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.llm.generate(**enc, max_new_tokens=16, do_sample=False,
                                    pad_token_id=self.mtok.eos_token_id,
                                    eos_token_id=self.mtok.eos_token_id)
        return self.mtok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def handle(self, messages):
        tool, raw = self.tool_call(messages)
        if not isinstance(tool, dict) or tool.get("tool") != "search_memory":
            return {"tool": tool, "raw": raw, "answer": None, "notes": []}
        args = tool.get("args", {})
        query = args.get("query", "")
        project = args.get("project", "")
        ids, mode = self.search(query, project)
        ids = self.rerank(query, ids, self.top_k)
        notes = [extract_note(self.docs[int(i)]) for i in ids]
        canon = canonical_from_note(notes[0]) or query
        answer = self.answer(canon, ids)
        return {"tool": tool, "raw": raw, "mode": mode, "query": canon,
                "answer": answer, "notes": notes}
