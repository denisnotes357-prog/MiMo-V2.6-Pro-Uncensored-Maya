# needle.py CHARS[,CHARS...] - a fact hidden mid-context ("needle") and asked for at the end: long-context attention
# correctness; prints prompt tokens, read and answer speed, the answer and whether it holds the code
import glob, os
import json,urllib.request,sys
src=("".join(open(p, encoding="utf-8").read() for p in sorted(glob.glob("docs/*.md")) + ["README.project-maya.md"]))*12   # public docs as filler
NEEDLE="\n\nВАЖНО: секретный код для проверки памяти — 7341-ОМУЛЬ-92. Запомни его.\n\n"
for n in [int(x) for x in sys.argv[1].split(",")]:
    body_txt=src[:n//2]+NEEDLE+src[n//2:n]
    q="Ниже длинный текст. Прочитай его.\n\n"+body_txt+"\n\nВопрос: какой секретный код для проверки памяти упомянут в тексте? Ответь только кодом."
    body={"model":"mimo-pro","messages":[{"role":"user","content":q}],"max_tokens":40,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=3600))
    ti=d["timings"]; a=d['choices'][0]['message']['content'].strip()
    print(f"needle ctx {ti['prompt_n']}: read {ti['prompt_per_second']:.0f} tok/s | answer {a!r} | {'OK' if '7341-ОМУЛЬ-92' in a else 'MISS'}", flush=True)
