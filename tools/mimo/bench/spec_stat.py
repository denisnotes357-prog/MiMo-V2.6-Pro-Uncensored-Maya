# spec_stat.py N - same process A/B: spec sampling (no bias) vs plain sampling (a -0.001 logit_bias turns spec off);
# digit frequencies of "30 random digits" past the first digit, chi-square between the arms
import json,urllib.request,sys,re,math,time
N=int(sys.argv[1]) if len(sys.argv)>1 else 25
Q="Write 30 random digits from 0 to 9, separated by spaces. Output only the digits, nothing else."
def run(seed,bias):
    body={"model":"mimo-pro","messages":[{"role":"user","content":Q}],"max_tokens":90,"temperature":1.0,"top_p":1.0,"seed":seed,
          "chat_template_kwargs":{"enable_thinking":False}}
    if bias: body["logit_bias"]={"100000":-0.001}
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=600))
    return re.findall(r"\d",d['choices'][0]['message']['content'])[1:], d['timings']['predicted_per_second']
res={}
for arm,bias,base in (("spec",False,1),("plain",True,1001)):
    cnt=[0]*10; tps=[]; seqs=set()
    for i in range(N):
        ds,v=run(base+i,bias); tps.append(v); seqs.add("".join(ds))
        for c in ds: cnt[int(c)]+=1
    res[arm]=cnt; print(arm, "tok/s %.1f"%(sum(tps)/len(tps)), "distinct seqs", len(seqs), "counts", cnt, flush=True)
a,b=res["spec"],res["plain"]; na,nb=sum(a),sum(b); chi=0
for x,y in zip(a,b):
    if x+y==0: continue
    e1=(x+y)*na/(na+nb); e2=(x+y)*nb/(na+nb); chi+=(x-e1)**2/e1+(y-e2)**2/e2
print("chi2 %.1f (df 9: 5%% critical 16.9, 1%% 21.7)"%chi)
