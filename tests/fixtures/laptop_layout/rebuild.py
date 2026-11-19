import sys,os,csv,shutil
H=os.path.dirname(os.path.abspath(__file__));O=sys.argv[1]
L=open(os.path.join(O,'LINKS.txt'),'w') if os.makedirs(O,exist_ok=True) is None else None
for r in csv.DictReader(open(os.path.join(H,'tree.tsv')),delimiter='\t'):
    p=os.path.join(O,r['path'])
    if r['link_target']: L.write(f"{r['path']}\t{r['link_target']}\n")
    if r['type'].startswith('d'): os.makedirs(p,exist_ok=True); continue
    os.makedirs(os.path.dirname(p),exist_ok=True); m=os.path.join(H,'meta',r['path'])
    shutil.copy(m,p) if os.path.exists(m) else open(p,'wb').close()
