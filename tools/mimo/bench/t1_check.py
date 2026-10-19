# t1_check.py OUT [reps] - t=1 answers (400 tok, no thinking) to 4 prompts, speed + acceptance, full texts saved to OUT
import glob, os
import json,urllib.request,sys
LOG=os.environ.get("MAYA_ENGINE_LOG", "")   # the engine log (draft-acceptance lines); optional
def last(p):
    if not LOG or not os.path.exists(LOG): return ""
    ls=[l for l in open(LOG,errors='ignore') if p in l]; return ls[-1].split(":",2)[-1].strip() if ls else ""
PR=[("ru","Напиши подробный рассказ (около 300 слов) о путешествии на Байкал зимой."),
    ("code","Write a Python implementation of an LRU cache class with get/put, docstrings and unit tests."),
    ("en","Explain how a CPU cache hierarchy works, with concrete latency numbers."),
    ("ru2","Объясни простыми словами, как работает квантование нейросетей и зачем оно нужно.")]
reps=int(sys.argv[2]) if len(sys.argv)>2 else 2; bias=len(sys.argv)>3 and sys.argv[3]=="plain"; out=[]; sums={}
for r in range(reps):
    for name,q in PR:
        body={"model":"mimo-pro","messages":[{"role":"user","content":q}],"max_tokens":400,"temperature":1.0,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":False}}
        if bias: body["logit_bias"]={"100000":-0.001}
        d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
        v=d['timings']['predicted_per_second']; sums.setdefault(name,[]).append(v)
        out.append({"p":name,"tps":v,"acc":last('drafts accepted by depth')[:70],"text":d['choices'][0]['message']['content']})
        print(f"{name}: {v:.1f} tok/s | {out[-1]['acc']}", flush=True)
print("mean:", {k: round(sum(v)/len(v),1) for k,v in sums.items()})
json.dump(out,open(sys.argv[1],"w"),ensure_ascii=False,indent=1)
