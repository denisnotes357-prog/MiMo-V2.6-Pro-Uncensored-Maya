import glob, os
import json,urllib.request,sys
LOG=os.environ.get("MAYA_ENGINE_LOG", "")   # the engine log (draft-acceptance lines); optional
def last(p):
    if not LOG or not os.path.exists(LOG): return ""
    ls=[l for l in open(LOG,errors='ignore') if p in l]; return ls[-1].split(":",2)[-1].strip() if ls else ""
P="Напиши подробный рассказ (около 300 слов) о путешествии на Байкал зимой."
C="Write a Python implementation of an LRU cache class with get/put, docstrings and unit tests."
for name,q,t in (("ru t=0",P,0),("code t=0",C,0),("ru t=1",P,1.0),("code t=1",C,1.0)):
    body={"model":"mimo-pro","messages":[{"role":"user","content":q}],"max_tokens":400,"temperature":t,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":False}}
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
    ti=d["timings"]
    print(f"{name}: {ti['predicted_per_second']:.1f} tok/s | {last('drafts accepted by depth')[:75]} | {d['choices'][0]['message']['content'][:60].replace(chr(10),' ')}", flush=True)
