import re, base64, html, json, difflib, time, urllib.request, urllib.parse
x = open("sub_v1/word/document.xml", encoding="utf8").read()
instr = html.unescape("".join(re.findall(r"<w:instrText[^>]*>(.*?)</w:instrText>", x, re.S)))
recs = re.findall(r"<record>.*?</record>", instr, re.S)
for b in re.findall(r"<w:fldData[^>]*>(.*?)</w:fldData>", x, re.S):
    recs += re.findall(r"<record>.*?</record>", html.unescape(base64.b64decode(re.sub(r"\s", "", b)).decode("utf8", "ignore")), re.S)
def strip(s): return re.sub(r"<[^>]+>", "", s).strip()
EN = {}
for r in recs:
    t = re.search(r"<titles>\s*<title>(.*?)</title>", r, re.S)
    a = re.search(r"<contributors>\s*<authors>(.*?)</authors>", r, re.S)   # primary authors only
    if t: EN[strip(t.group(1))] = [strip(v) for v in re.findall(r"<author>(.*?)</author>", a.group(1), re.S)] if a else []
def first_last(n):
    n = re.sub(r"\s+", " ", n).strip()
    if "," in n:
        fam, giv = [p.strip() for p in n.split(",", 1)]
        return f"{giv} {fam}".strip()
    return n
def initials_only(n): return bool(re.fullmatch(r"(?:[A-Z]\.(?:-[A-Z]\.)?\s*)+[^\s]+", n.strip())) or bool(re.match(r"^[^,]+,\s*(?:[A-Z]\.\s*)+$", n.strip()))
def crossref(doi):
    it = json.load(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi),
          headers={"User-Agent": "doi-check/0.1"}), timeout=30))["message"]
    time.sleep(1)
    return [f"{a.get('given','')} {a.get('family','')}".strip() for a in it.get("author", []) if a.get("family")]
def join(names):
    return names[0] if len(names) == 1 else (" and ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + ", and " + names[-1])
def norm(s): return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
refs = open("ms/refs_acm.txt").read().strip().split("\n")
out, report = [], []
CORP = ("European Parliament", "Centers for Disease", "Article 29", "Anonymous.")
for i, r in enumerate(refs):
    m = re.match(r"^(.*?)\. ((?:19|20)\d\d)\. (.*)$", r)
    au_old, year, rest = m.group(1), m.group(2), m.group(3)
    if r.startswith(CORP): out.append(r); report.append((i + 1, "kept (organisation)")); continue
    title = rest.split(". ")[0]
    best = max(EN, key=lambda t: difflib.SequenceMatcher(None, norm(t), norm(title)).ratio())
    sim = difflib.SequenceMatcher(None, norm(best), norm(title)).ratio()
    names = [first_last(n) for n in EN[best]] if sim > 0.8 else []
    src = "EndNote"
    if not names or any(initials_only(n) for n in EN.get(best, [])):
        doi = re.search(r"https://doi.org/(10\.\S+)", r)
        if doi and not doi.group(1).startswith("10.48550"):
            cr = crossref(doi.group(1))
            if cr and len(cr) == len(names or cr): names, src = cr, "Crossref"
    if not names: out.append(r); report.append((i + 1, "UNCHANGED (no match)")); continue
    n_old = len(re.split(r",\s*(?:and\s+)?|\s+and\s+", au_old))
    out.append(f"{join(names)}. {year}. {rest}")
    report.append((i + 1, f"{src}: {len(names)} authors (was {n_old})", join(names)[:90]))
open("ms/refs_acm_fullnames.txt", "w").write("\n".join(out) + "\n")
for rr in report: print(*rr)
