import json, os, time, sys
from collections import defaultdict, Counter
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

BASE = "/home/daniel/tnc"
CORPUS = os.path.join(BASE, "corpus.jsonl")
OUT = os.path.join(BASE, "work")
os.makedirs(OUT, exist_ok=True)

CHUNK = 128
SEED = 20261030

def floor_byte_to_char_map(text, targets):
    """Map byte offsets to char offsets, flooring to the start of the char
    containing that byte. targets must be non-negative ints."""
    targets = sorted(set(int(t) for t in targets))
    res = {}
    b = 0
    ci = 0
    ti = 0
    while ti < len(targets) and targets[ti] <= 0:
        res[targets[ti]] = 0
        ti += 1
    for ci, ch in enumerate(text):
        bl = 1 if ord(ch) < 128 else len(ch.encode("utf-8"))
        b += bl
        while ti < len(targets) and targets[ti] <= b:
            res[targets[ti]] = ci + 1 if targets[ti] == b else ci
            ti += 1
    while ti < len(targets):
        res[targets[ti]] = len(text)
        ti += 1
    return res

def main():
    t_start = time.time()
    tok = Tokenizer.from_pretrained("gpt2")
    tok.no_truncation()
    tok.no_padding()
    V = tok.get_vocab_size()
    blank_vocab = np.zeros(V, dtype=bool)
    for i in range(V):
        s = tok.decode([i], skip_special_tokens=False)
        if s.strip() == "":
            blank_vocab[i] = True
    print(f"vocab={V} blank_token_types={int(blank_vocab.sum())}", flush=True)

    test = "héllo\nworld"
    enc_test = tok.encode(test)
    maxoff = max(e for _, e in enc_test.offsets)
    byte_mode = maxoff > len(test)
    print(f"offset_mode={'byte' if byte_mode else 'char'} test_offsets={enc_test.offsets} "
          f"len_chars={len(test)} len_bytes={len(test.encode('utf-8'))}", flush=True)

    channels = defaultdict(list)
    n_lines = 0
    with open(CORPUS, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            channels[o["ch"]].append(o["t"])
            n_lines += 1
    print(f"messages={n_lines} channels={len(channels)} load_s={time.time()-t_start:.1f}", flush=True)

    all_chunks = []
    all_tids = []
    all_flags = []
    raw_total = blank_total = content_total = 0
    leading_blanks_total = 0
    dropped_content_total = 0
    dropped_blank_total = 0
    flag_counter = Counter()
    per_channel = {}
    t_tok = time.time()

    for chi, (ch, msgs) in enumerate(tqdm(channels.items(), desc="channels")):
        text = "".join(m + "\n" for m in msgs)
        enc = tok.encode(text)
        offsets = enc.offsets
        ids = np.asarray(enc.ids, dtype=np.uint16)
        n_raw = len(ids)
        is_blank = blank_vocab[ids]
        n_blank = int(is_blank.sum())
        content = np.nonzero(~is_blank)[0]
        n_content = len(content)
        raw_total += n_raw
        blank_total += n_blank
        content_total += n_content

        if n_content == 0:
            leading_blanks_total += n_raw
            per_channel[ch] = {"raw": n_raw, "blank": n_blank, "content": 0, "chunks": 0}
            continue

        leading_blanks_total += int(content[0])
        # blank-run flag on each content token = blanks immediately following it
        run = np.zeros(n_raw, dtype=np.uint8)
        for k in range(n_content - 1, -1, -1):
            c = int(content[k])
            nxt = int(content[k + 1]) if k + 1 < n_content else n_raw
            run[c] = min(3, nxt - c - 1)

        n_full = n_content // CHUNK
        dropped = n_content - n_full * CHUNK
        dropped_content_total += dropped
        if n_full == 0:
            dropped_blank_total += n_blank
            per_channel[ch] = {"raw": n_raw, "blank": n_blank, "content": n_content, "chunks": 0}
            continue

        total_bytes = len(text.encode("utf-8"))
        total_len = total_bytes if byte_mode else len(text)
        needed = set()
        for j in range(n_full):
            c0 = int(content[j * CHUNK])
            needed.add(int(offsets[c0][0]))
            nxt_k = (j + 1) * CHUNK
            if nxt_k < n_content:
                needed.add(int(offsets[int(content[nxt_k])][0]))
            else:
                needed.add(total_len)
        if byte_mode:
            bmap = floor_byte_to_char_map(text, needed)
            def to_char(b):
                return bmap[int(b)]
        else:
            def to_char(b):
                return int(b)

        for j in range(n_full):
            c0 = int(content[j * CHUNK])
            c1 = int(content[j * CHUNK + CHUNK - 1])
            b_start = int(offsets[c0][0])
            nxt_k = (j + 1) * CHUNK
            b_end = int(offsets[int(content[nxt_k])][0]) if nxt_k < n_content else total_len
            s = to_char(b_start)
            e = to_char(b_end)
            chunk_text = text[s:e]
            tids = ids[content[j * CHUNK:(j + 1) * CHUNK]]
            fl = run[content[j * CHUNK:(j + 1) * CHUNK]]
            all_chunks.append({
                "channel": ch,
                "chunk_in_channel": j,
                "text": chunk_text,
                "char_start": s,
                "char_end": e,
            })
            all_tids.append(tids.copy())
            all_flags.append(fl.copy())
            for v in fl:
                flag_counter[int(v)] += 1
        per_channel[ch] = {"raw": n_raw, "blank": n_blank, "content": n_content, "chunks": n_full}

    t_tok_done = time.time()
    print(f"tokenize+chunk_s={t_tok_done-t_tok:.1f} chunks={len(all_chunks)} "
          f"raw={raw_total} blank={blank_total} content={content_total}", flush=True)

    tids = np.stack(all_tids, axis=0)
    flags = np.stack(all_flags, axis=0)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_chunks))
    chunks = [all_chunks[i] for i in perm]
    tids = tids[perm]
    flags = flags[perm]
    for pos, c in enumerate(chunks):
        c["pos"] = pos

    with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(os.path.join(OUT, "token_ids.npy"), tids)
    np.save(os.path.join(OUT, "flags.npy"), flags)

    stats = {
        "corpus": CORPUS,
        "messages": n_lines,
        "channels": len(channels),
        "raw_tokens": raw_total,
        "blank_tokens": blank_total,
        "content_tokens": content_total,
        "blank_fraction": blank_total / raw_total if raw_total else None,
        "leading_blanks_unassigned": leading_blanks_total,
        "content_tokens_in_full_chunks": int(len(chunks) * CHUNK),
        "content_tokens_dropped_in_partial_chunks": int(dropped_content_total),
        "blank_flag_distribution_content_tokens": {str(k): int(v) for k, v in sorted(flag_counter.items())},
        "chunk_size_content_tokens": CHUNK,
        "shuffle_seed": SEED,
        "n_chunks": len(chunks),
        "tokenizer": "gpt2",
        "tokenizer_vocab": int(V),
        "offset_mode": "byte" if byte_mode else "char",
        "per_channel": per_channel,
        "timing_s": {"load": t_tok - t_start, "tokenize_chunk": t_tok_done - t_tok, "total": time.time() - t_start},
    }
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: stats[k] for k in ["raw_tokens", "blank_tokens", "content_tokens",
          "blank_fraction", "n_chunks", "content_tokens_in_full_chunks",
          "content_tokens_dropped_in_partial_chunks"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
