# lookup_bench.py OUT - context-copying tasks (t=0): a code edit that repeats a file, a Russian proofreading pass, plus a
# free-writing control; tok/s, the server's lookup line, answers saved to OUT for a token-exact comparison
import glob, os
import json,urllib.request,sys,time
LOG=os.environ.get("MAYA_ENGINE_LOG", "")   # the engine log (draft-acceptance lines); optional
def last(p):
    if not LOG or not os.path.exists(LOG): return ""
    ls=[l for l in open(LOG,errors='ignore') if p in l]; return ls[-1].split(":",1)[-1].strip() if ls else "-"
code=open("./tools/mimo/mtp3_ref.py").read()
ru=("Байкал — самое глубокое озеро на планете и крупнейший природный резервуар пресной воды. Его возраст оценивают в "
    "25–35 миллионов лет, что делает его одним из древнейших озёр мира. Зимой поверхность озера покрываеться прозрачным "
    "льдом толщиной до двух метров, и сквозь него видно дно на глубине нескольких десятков метров. В озере обитает "
    "больше двух тысяч видов животных, большинство из которых эндемики, то есть не встречаються больше нигде. Самый "
    "известный из них — байкальская нерпа, единственный пресноводный тюлень. Летом вода прогревается только у берегов, "
    "а на глубине температура круглый год держиться около четырёх градусов. Каждый год сюда приезжают тысячи туристов, "
    "чтобы увидеть знаменитый лёд, покататься на коньках и сфотографировать торосы.")
T=[("code-edit","Here is a Python file:\n```python\n"+code+"```\nRename the inner function `run` to `run_block` everywhere "
                "(definition and calls). Output the complete modified file in one ```python block, nothing else.",1800),
   ("ru-proof","Исправь орфографические ошибки в тексте и выведи весь текст целиком без комментариев:\n\n"+ru,500),
   ("ru-free","Напиши подробный рассказ (около 300 слов) о путешествии на Байкал зимой.",400)]
res=[]
for name,q,mx in T:
    body={"model":"mimo-pro","messages":[{"role":"user","content":q}],"max_tokens":mx,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1800))
    time.sleep(0.3); ti=d["timings"]; c=d['choices'][0]['message']['content']; res.append(c)
    print(f"{name:9s}: {ti['predicted_n']:4d} tok {ti['predicted_per_second']:6.1f} tok/s | lookup: {last('prompt lookup')[:70]} | MTP {ti.get('draft_n_accepted')}/{ti.get('draft_n')}", flush=True)
json.dump(res,open(sys.argv[1],"w"),ensure_ascii=False,indent=0)
