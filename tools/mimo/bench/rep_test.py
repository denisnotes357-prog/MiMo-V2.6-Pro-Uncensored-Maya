#!/usr/bin/env python3
"""rep_test.py N TAG - N long Russian scenes at t=1/top_p .95; repeated-line metrics (what the book showed)."""
import collections, json, sys, time, urllib.request
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
P = ("Напиши сцену для кровавого экшн-хоррора, 1486 год, пыточный подвал инквизитора Крамера, допрос ведьмы, которая смеётся под пыткой. "
     "Стиль: кадры и крупные планы («Камера — крупно: ...»), настоящее время, короткие абзацы, много реплик. Объём — около 9000 знаков. Только текст сцены.")
def ask(seed):
    b = {"model": "mimo-pro", "messages": [{"role": "user", "content": P}], "max_tokens": 6000, "temperature": 1.0, "top_p": 0.95, "seed": seed}
    r = urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json"})
    d = json.load(OP.open(r, timeout=1800)); return d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"]
n, tag = int(sys.argv[1]), sys.argv[2]
for i in range(n):
    t0 = time.time(); txt, ntok = ask(1000 + i)
    L = [l.strip() for l in txt.splitlines() if len(l.strip()) > 12]
    c = collections.Counter(L); mx = max(c.values()) if c else 0
    dup = sum(v - 1 for v in c.values()) / max(1, len(L))
    lat = sum(ch.isascii() and ch.isalpha() for ch in txt); cyr = sum("а" <= ch.lower() <= "я" for ch in txt)
    print(json.dumps({"tag": tag, "latin_frac": round(lat / max(1, lat + cyr), 3), "seed": 1000 + i, "chars": len(txt), "tok": ntok, "lines": len(L), "max_rep": mx, "dup_frac": round(dup, 3),
                      "top": c.most_common(1)[0][0][:60] if c else "", "s": round(time.time() - t0)}, ensure_ascii=False), flush=True)
    open(f"rep-{tag}-{i}.txt", "w").write(txt)
