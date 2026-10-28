import json, re, time, urllib.parse, urllib.request, difflib
refs = open("refs_acm.txt").read().strip().split("\n")
def title_year(r):
    m = re.match(r"^(.*?)\. ((?:19|20)\d\d)\. (.*?)\. (?=\*|In \*|arXiv|Retrieved|Adopted|Software)", r)
    return (m.group(3), int(m.group(2)), m.group(1)) if m else (None, None, None)
def norm(s): return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
out = {}
for i, r in enumerate(refs):
    if "doi.org" in r or "arXiv:" in r or "Retrieved from" in r or "Adopted on" in r: continue
    t, y, au = title_year(r)
    if not t: print("PARSE FAIL", i, r[:80]); continue
    first_author = au.split(",")[0].split()[-1]
    q = urllib.parse.urlencode({"query.bibliographic": f"{t} {first_author}", "rows": 5})
    try:
        data = json.load(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works?" + q,
                         headers={"User-Agent": "doi-check/0.1"}), timeout=30))
    except Exception as e:
        print("ERR", i, e); continue
    best = None
    for it in data["message"]["items"]:
        ct = (it.get("title") or [""])[0]
        sim = difflib.SequenceMatcher(None, norm(t), norm(ct)).ratio()
        yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        if best is None or sim > best[0]: best = (sim, it["DOI"], ct, yr, it.get("type"), (it.get("container-title") or [""])[0])
    out[i] = dict(ref=r[:90], title=t, year=y, best=best)
    flag = "OK " if best and best[0] >= 0.95 and best[3] and abs(best[3] - y) <= 1 else "?? "
    print(flag, i + 1, f"sim={best[0]:.2f}", best[1], "|", best[3], "|", best[4], "|", best[2][:70], "|", best[5][:40])
    time.sleep(1)
json.dump(out, open("doi_lookup.json", "w"), indent=1)
