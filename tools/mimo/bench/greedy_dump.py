# greedy_dump.py OUT - t=0 answers (400 tokens, no thinking) to fixed prompts, for token-exact comparison between builds
import json,urllib.request,sys
PR=["Напиши подробный рассказ (около 300 слов) о путешествии на Байкал зимой.",
    "Write a Python implementation of an LRU cache class with get/put, docstrings and unit tests.",
    "Explain how a CPU cache hierarchy works, with concrete latency numbers.",
    "Объясни простыми словами, как работает квантование нейросетей и зачем оно нужно."]
out=[]
for q in PR:
    body={"model":"mimo-pro","messages":[{"role":"user","content":q}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
    out.append(d['choices'][0]['message']['content']); print(f"{d['timings']['predicted_per_second']:.1f} tok/s", flush=True)
json.dump(out,open(sys.argv[1],"w"),ensure_ascii=False,indent=0)
