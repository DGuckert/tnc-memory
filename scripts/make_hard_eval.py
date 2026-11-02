import json, os, random, re

BASE = "/home/daniel/tnc"
WORK = os.path.join(BASE, "work")
TRAIN = "/home/daniel/tnc-repo/training"
NOUNS = ["access code", "invoice number", "serial number", "passphrase", "license key", "tracking ID"]
SEED = 9901

def make_names(n, rng):
    onsets = ["b","d","f","g","k","l","m","n","p","r","s","t","v","z","br","dr","gr","kr","tr","vr","sk","st","th","zh","qu"]
    vowels = ["a","e","i","o","u","ae","ia","ou"]
    codas = ["","n","r","l","s","th","x","nd","rk","st","m","v"]
    seen=set(); out=[]
    while len(out)<n:
        s=rng.choice(onsets)+rng.choice(vowels)+rng.choice(codas)
        if rng.random()<0.5:
            s+=rng.choice(onsets)+rng.choice(vowels)+rng.choice(codas)
        s=s.capitalize()
        if s not in seen and len(s)>=4:
            seen.add(s); out.append(s)
    return out

def make_codes(n, rng):
    alphabet="ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; seen=set(); out=[]
    while len(out)<n:
        s="".join(rng.choice(alphabet) for _ in range(7)); s=s[:3]+"-"+s[3:]
        if s not in seen: seen.add(s); out.append(s)
    return out

def fact(p):
    return f"Note to self: the {p['noun']} for project {p['name']} is {p['code']}."

def typo(name, rng, known):
    vs=[name[:-2]+name[-1]+name[-2], name[:-1], name+name[-1], name[0]+name[2:]]
    rng.shuffle(vs)
    for v in vs:
        if v!=name and v not in known and len(v)>=4:
            return v
    return name

def main():
    known=set()
    for fn in ["train_sft.jsonl","val_sft.jsonl","test_sft.jsonl","train_reranker.jsonl","val_reranker.jsonl","test_reranker.jsonl"]:
        for line in open(os.path.join(TRAIN,fn),encoding="utf-8"):
            known |= set(re.findall(r"project ([A-Za-z0-9]+)", line))
    bench=json.load(open(os.path.join(WORK,"needles_10485760.json"),encoding="utf-8"))
    known |= {f["name"] for f in bench["facts"]}
    rng=random.Random(SEED)
    names=[]
    while len(names)<80:
        for n in make_names(200,rng):
            if n not in known and n not in names:
                names.append(n)
                if len(names)>=80: break
    codes=[c for c in make_codes(100,rng)]
    projects=[{"name":n,"code":codes[i],"noun":NOUNS[i%len(NOUNS)]} for i,n in enumerate(names)]

    # real chatter lines as noise
    pool=[]
    with open(os.path.join(WORK,"chunks.jsonl"),encoding="utf-8") as f:
        for line in f:
            text=json.loads(line)["text"]
            for s in text.splitlines():
                s=s.strip()
                if 20<=len(s)<=160 and not re.search(r"note to self|project |access code|invoice number|serial number|passphrase|license key|tracking id", s, re.I):
                    pool.append(s)
    rng.shuffle(pool)
    print("pool",len(pool),flush=True)

    acks=["yeah","ok","lol","makes sense","true","same","got it","fair"]
    intros=["btw {name} is live now.","we're doing {name} this week.","{name} is the active project.","I moved {name} to the top of the list.","{name} just came in."]
    explicit=["can you pull the {noun} for {name}?","I need the {noun} for the {name} job.","what did we set {name}'s {noun} to?","do you have the {noun} on file for {name}?","what's the {noun} for {name} again?"]
    pronoun=["what's the {noun} for it?","do you have its {noun}?","what's the {noun} for that project?","can you get the {noun} for it?"]
    other=["and the {noun} for the other one?","what about the {noun} for the other project?"]
    new=["what's the {noun} for the new one?","and the {noun} for the new project?"]
    rows=[]
    for i in range(100):
        r=rng.random()
        if r<0.05:
            target={"tool":"ask_clarification","args":{"question":"Which project do you mean?"}}
            turns=[{"role":"system","content":"You are a memory-aware assistant. Resolve the user's project reference from the conversation. If a project is clear, call search_memory with the canonical project name and a self-contained query. If no project is clear, call ask_clarification. For small talk, call no_tool. Reply with JSON only."}]
            p=rng.choice(projects)
            turns.append({"role":"user","content":rng.choice(pronoun).format(noun=p["noun"])})
            turns.append({"role":"assistant","content":json.dumps(target,separators=(",",":"))})
            rows.append({"id":f"hard-tool-{i:04d}","task":"tool_call","messages":turns,"target":target,"meta":{"case":"clarify"}})
            continue
        if r<0.10:
            target={"tool":"no_tool","args":{}}
            turns=[{"role":"system","content":"You are a memory-aware assistant. Resolve the user's project reference from the conversation. If a project is clear, call search_memory with the canonical project name and a self-contained query. If no project is clear, call ask_clarification. For small talk, call no_tool. Reply with JSON only."},
                   {"role":"user","content":rng.choice(["yo","thanks","talk later","how's it going?"])},
                   {"role":"assistant","content":json.dumps(target,separators=(",",":"))}]
            rows.append({"id":f"hard-tool-{i:04d}","task":"tool_call","messages":turns,"target":target,"meta":{"case":"small_talk"}})
            continue

        target_p=rng.choice(projects)
        others=rng.sample([p for p in projects if p["name"]!=target_p["name"]], rng.choice([0,1,2]))
        order=(others+[target_p])
        rng.shuffle(order)
        case=rng.choices(["explicit","pronoun","other","new"],[55,25,10,10],k=1)[0]
        if case in ("pronoun","new"):
            order=[p for p in order if p["name"]!=target_p["name"]]+[target_p]
        elif case=="other":
            order=[target_p]+[p for p in order if p["name"]!=target_p["name"]]
            if len(order)<2:
                other_p=rng.choice([p for p in projects if p["name"]!=target_p["name"]])
                order=[target_p,other_p]
        turns=[{"role":"system","content":"You are a memory-aware assistant. Resolve the user's project reference from the conversation. If a project is clear, call search_memory with the canonical project name and a self-contained query. If no project is clear, call ask_clarification. For small talk, call no_tool. Reply with JSON only."}]
        for _ in range(rng.choice([2,3,4])):
            turns.append({"role":"user","content":rng.choice(pool)})
            turns.append({"role":"assistant","content":rng.choice(acks)})
        for p in order:
            turns.append({"role":"user","content":rng.choice(intros).format(name=p["name"])})
            turns.append({"role":"assistant","content":rng.choice(acks)})
            if rng.random()<0.4:
                turns.append({"role":"user","content":rng.choice(pool)})
                turns.append({"role":"assistant","content":rng.choice(acks)})
        name=target_p["name"]
        if case=="explicit" and rng.random()<0.2:
            name=typo(name,rng,known)
        if case=="explicit": q=rng.choice(explicit).format(noun=target_p["noun"],name=name)
        elif case=="pronoun": q=rng.choice(pronoun).format(noun=target_p["noun"])
        elif case=="other": q=rng.choice(other).format(noun=target_p["noun"])
        else: q=rng.choice(new).format(noun=target_p["noun"])
        target={"tool":"search_memory","args":{"project":target_p["name"],"query":f"what is the {target_p['noun']} for project {target_p['name']}?"}}
        turns.append({"role":"user","content":q})
        turns.append({"role":"assistant","content":json.dumps(target,separators=(",",":"))})
        rows.append({"id":f"hard-tool-{i:04d}","task":"tool_call","messages":turns,"target":target,"meta":{"case":case,"target_project":target_p["name"],"noun":target_p["noun"]}})
    with open(os.path.join(WORK,"hard_eval.jsonl"),"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    meta={"n":len(rows),"cases":{c:sum(1 for r in rows if r["meta"]["case"]==c) for c in set(r["meta"]["case"] for r in rows)},"names":names}
    json.dump(meta,open(os.path.join(WORK,"hard_eval_meta.json"),"w"),indent=1)
    print(meta["cases"],flush=True)

if __name__=="__main__":
    main()
