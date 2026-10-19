# short_turn.py OUT - short prompts' reading: fresh short prompts and a chat's short second turn (t=0); saves answers
import json,urllib.request,sys
def ask(msgs,mx=120):
    body={"model":"mimo-pro","messages":msgs,"max_tokens":mx,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
res=[]
for q in ["Привет! Как дела?","What is the capital of Australia?","Назови три планеты.","2+2=?","Translate 'good morning' to French."]:
    d=ask([{"role":"user","content":q}]); ti=d["timings"]; c=d['choices'][0]['message']['content']; res.append(c)
    print(f"fresh {ti['prompt_n']:>3} tok: read {ti['prompt_per_second']:6.0f} tok/s ({ti['prompt_ms']:.0f} ms) | gen {ti['predicted_per_second']:.1f} | {c[:70]!r}", flush=True)
m=[{"role":"user","content":"Расскажи коротко (5 предложений) про озеро Байкал."}]
d=ask(m,200); a=d['choices'][0]['message']['content']; m.append({"role":"assistant","content":a})
for q in ["А какая там глубина?","Сколько там видов рыб? Ответь одним предложением.","Спасибо! А когда лучше ехать?"]:
    m.append({"role":"user","content":q}); d=ask(m,150); ti=d["timings"]; c=d['choices'][0]['message']['content']; res.append(c)
    print(f"turn  {ti['prompt_n']:>3} tok: read {ti['prompt_per_second']:6.0f} tok/s ({ti['prompt_ms']:.0f} ms) | gen {ti['predicted_per_second']:.1f} | MTP {ti.get('draft_n_accepted')}/{ti.get('draft_n')} | {c[:60]!r}", flush=True)
    m.append({"role":"assistant","content":c})
json.dump(res,open(sys.argv[1],"w"),ensure_ascii=False,indent=0)
