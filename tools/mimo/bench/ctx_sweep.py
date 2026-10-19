import glob, os
import json,urllib.request,time,sys,re
LOG=os.environ.get("MAYA_ENGINE_LOG", "")   # the engine log (draft-acceptance lines); optional
def last(pat):
    if not LOG or not os.path.exists(LOG): return ""
    ls=[l for l in open(LOG,errors='ignore') if pat in l]; return ls[-1].strip() if ls else ""
src=("".join(open(p, encoding="utf-8").read() for p in sorted(glob.glob("docs/*.md")) + ["README.project-maya.md"]))*12   # public docs as filler
for n in [int(x) for x in sys.argv[1].split(",")]:
    q="Ниже текст для контекста. После него напиши связный рассказ (около 250 слов) о путешествии на Байкал зимой — к тексту он не относится.\n\n"+src[:n]
    body={"model":"mimo-pro","messages":[{"role":"user","content":q}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=3600))
    ti=d["timings"]; time.sleep(0.5); st=last("glm stat: decode")
    m=re.search(r"decode ([0-9.]+) ms/tok",st)
    print(f"ctx {ti['prompt_n']:>6}: read {ti['prompt_per_second']:6.0f} tok/s | gen {ti['predicted_n']} tok {ti['predicted_per_second']:5.1f} tok/s | MTP {ti.get('draft_n_accepted')}/{ti.get('draft_n')} | {d['choices'][0]['message']['content'][:90].replace(chr(10),' ')}", flush=True)
