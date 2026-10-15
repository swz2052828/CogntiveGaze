import re
L=open("/springbrook/share/eng/esrpxk/results/paper/CCF_PASTE_ORDER.md").read().splitlines()
hdr=[i for i,l in enumerate(L) if re.match(r"^##+ ",l)]
def sect(p):
    i=[k for k in hdr if L[k].lstrip("#").strip().startswith(p)][0]
    return i,min([k for k in hdr if k>i] or [len(L)])
for name,pref in [("abstract","1. Abstract"),("intro","2. Introduction"),("contrib","Contributions"),
                  ("related","3. Related Work"),("threat","7. Threat Model"),("leak","8. Identity Leakage"),
                  ("pre","9-pre"),("c9c","9c. Accompanying"),("c9cter","9c-ter"),("c9cbis","9c-bis"),
                  ("disc","9.5 Discussion"),("limits","10. Limitations")]:
    i,j=sect(pref)
    open(name+".md","w").write("\n".join(re.sub(r"^> ?","",l) for l in L[i:j] if l.startswith(">"))+"\n")
