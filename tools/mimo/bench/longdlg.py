# longdlg.py CHARS - a long context, then two DIFFERENT follow-up questions on it: the second should restore a checkpoint
# and read only its new tokens; prints new tokens read, read time, generation speed (no-copy tasks: honest long-ctx speed)
import glob, os
import json,urllib.request,sys
src=("".join(open(p, encoding="utf-8").read() for p in sorted(glob.glob("docs/*.md")) + ["README.project-maya.md"]))*12   # public docs as filler
n=int(sys.argv[1])
q="Ниже текст для контекста. Прочитай его.\n\n"+src[:n]
for follow in ["Write a short English poem about the sea (about 150 words). Do not quote the text above.",
               "Теперь напиши короткую сказку про лису и ворону (около 150 слов), своими словами.",
               "Напиши три совета, как зимой фотографировать лёд, коротко."]:
    m=[{"role":"user","content":q},{"role":"assistant","content":"Хорошо."},{"role":"user","content":follow}]
    body={"model":"mimo-pro","messages":m,"max_tokens":250,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=3600))
    ti=d["timings"]; c=d['choices'][0]['message']['content']
    print(f"longdlg {n} chars: read {ti['prompt_n']} new tok in {ti['prompt_ms']:.0f} ms | gen {ti['predicted_n']} tok {ti['predicted_per_second']:.1f} tok/s | MTP {ti.get('draft_n_accepted')}/{ti.get('draft_n')} | {c[:70]!r}", flush=True)
