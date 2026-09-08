# -*- coding: utf-8 -*-
"""Paginate ALL competition submissions, filter today's CD campaign, rank by private.
Run when scores have landed:  python analyze_private_campaign.py
"""
import os, io, sys, re, json, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi
# Kaggle credentials are read from the environment (KAGGLE_USERNAME / KAGGLE_KEY)
# or ~/.kaggle/kaggle.json. Never hard-code them.
assert os.environ.get("KAGGLE_KEY"), "Set KAGGLE_USERNAME and KAGGLE_KEY in your environment."
api=KaggleApi(); api.authenticate()
COMP="ai-agent-security-multi-step-tool-attacks"
rows=[]
for pg in range(1,12):
    try: subs=api.competition_submissions(COMP, page=pg)
    except TypeError: subs=api.competition_submissions(COMP)
    if not subs: break
    for s in subs:
        rows.append(dict(date=str(getattr(s,'date','')), desc=(getattr(s,'description','') or ''),
                         pub=getattr(s,'publicScore',None), pri=getattr(s,'privateScore',None),
                         status=str(getattr(s,'status',''))))
    if len(subs)<20: break
# dedup by (date,desc)
seen=set(); uniq=[]
for r in rows:
    k=(r['date'],r['desc'][:40])
    if k in seen: continue
    seen.add(k); uniq.append(r)
today=[r for r in uniq if str(r['date']).startswith('2026-09-05')]
def fl(x):
    try: return float(x)
    except: return None
scored=[r for r in today if fl(r['pri']) is not None]
print(f"fetched {len(uniq)} unique subs; today={len(today)} scored={len(scored)} pending={len(today)-len(scored)}")
scored.sort(key=lambda r:-(fl(r['pri']) or -1))
print("\n=== TOP private (today's campaign) ===")
for r in scored[:25]:
    tag=(re.search(r'(jed-[a-z0-9-]+)', r['desc']) or re.search(r'(CD [^:]+)', r['desc']))
    tag=tag.group(1) if tag else r['desc'][:40]
    print(f"  priv={fl(r['pri']):>8.3f}  pub={fl(r['pub']) if fl(r['pub']) is not None else 0:>8.3f}  {tag}")
# save
with open("campaign_scores.csv","w",newline='',encoding='utf-8') as fh:
    w=csv.writer(fh); w.writerow(["date","pub","pri","desc"])
    for r in today: w.writerow([r['date'],r['pub'],r['pri'],r['desc']])
print("\nsaved campaign_scores.csv")