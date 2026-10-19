import json,urllib.request,sys,re,collections
P="Напиши длинную сказку (около 1000 слов) про трёх ведьм и кита, которые оказались в современной Москве. Живой язык, диалоги, без повторов."
for i in range(int(sys.argv[1])):
    body={"model":"mimo-pro","messages":[{"role":"user","content":P}],"max_tokens":1800,"temperature":1.0,"top_p":0.95,"seed":100+i,"chat_template_kwargs":{"enable_thinking":False}}; body.update(json.loads(sys.argv[2]) if len(sys.argv)>2 else {})
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
    c=d['choices'][0]['message']['content']; w=c.split()
    # repetition score: share of repeated 8-word shingles
    sh=[" ".join(w[j:j+8]) for j in range(max(0,len(w)-8))]; cnt=collections.Counter(sh); rep=sum(v-1 for v in cnt.values() if v>1)/max(1,len(sh))
    print(f"run {i}: {d['timings']['predicted_n']} tok {d['timings']['predicted_per_second']:.0f} tok/s finish={d['choices'][0]['finish_reason']} rep8={rep:.2f}")
    print("   START:", c[:200].replace("\n"," ")); print("   END:  ", c[-300:].replace("\n"," "))
