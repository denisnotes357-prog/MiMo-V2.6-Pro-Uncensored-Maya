# stop_stat.py N - early-stop rate of "30 random digits": spec sampling vs plain (-0.001 logit_bias), same process
import json,urllib.request,re,sys
N=int(sys.argv[1]); Q="Write 30 random digits from 0 to 9, separated by spaces. Output only the digits, nothing else."
for arm,bias,base in (("spec",False,5000),("plain",True,9000)):
    hist={}
    for i in range(N):
        body={"model":"mimo-pro","messages":[{"role":"user","content":Q}],"max_tokens":90,"temperature":1.0,"top_p":1.0,"seed":base+i,"chat_template_kwargs":{"enable_thinking":False}}
        if bias: body["logit_bias"]={"100000":-0.001}
        d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=600))
        n=len(re.findall(r"\d",d['choices'][0]['message']['content'])); hist[n]=hist.get(n,0)+1
    print(arm, "digits->count", dict(sorted(hist.items())), "early", sum(v for k,v in hist.items() if k<30), flush=True)
