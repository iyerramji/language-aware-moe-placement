from common import *
from e_language_families import FAMILY
import csv
p,q,langs,m=data();folder=OUT/'E'
with (folder/'table.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==120;assert len({(x['gpus'],x['mix'],x['policy'],x['R']) for x in rows})==120
plans=json.loads((folder/'plans.json').read_text());assert len(plans)==40
for key,plan in plans.items():
 g=int(key.split('_')[0]);groups=plan['groups'];where=np.array(plan['layout']);assert sorted(sum(groups,[]))==sorted(langs);assert len(groups)==g;assert where.shape==(48,128)
 for l in range(48):assert all((where[l]==i).sum()==128//g for i in range(g))
 if key.endswith('_family'):
  assert groups[0]==['en'];assert all(len({FAMILY[x] for x in z})==1 for z in groups)
for row in rows:
 assert np.isfinite(float(row['local_share']));assert 0<=float(row['local_share'])<=1;assert float(row['imbalance_mean'])>=1-1e-9
 if row['policy']=='language_blind' and row['R']=='0':assert abs(float(row['local_share'])-1/int(row['gpus']))<1e-10
 # Recompute every saved measurement from fixed serialized plans and masks.
 key=f"{row['gpus']}_{row['mix']}_{row['policy']}";plan=plans[key];groups=[[langs.index(l) for l in z] for z in plan['groups']];w=np.array([plan['fit_mix'][l] for l in langs]);where=np.array(plan['layout']);ex=copies(p,w,groups,where,int(row['R']),blind=row['policy']=='language_blind');met=evaluate(q,w,groups,where,ex,blind=row['policy']=='language_blind');assert abs(met['local_share']-float(row['local_share']))<1e-12
with (folder/'shuffle-control.csv').open() as f:shuffled=list(csv.DictReader(f))
assert len(shuffled)==120
for row in shuffled:
 labels=row['pseudo_ia'].split(',');assert len(labels)==5;assert set(labels)!={l for l in langs if FAMILY[l]=='Indo-Aryan'}
with (OUT/'F/table.csv').open() as f:frows=list(csv.DictReader(f))
assert len(frows)==12
for key,b in json.loads((OUT/'F/allocations.json').read_text()).items():assert np.all(np.array(b).sum(0)==96)
# Mentor baseline files remain byte-identical to Git.
import subprocess
subprocess.run(['git','diff','--exit-code','--','analyze.py','results/exp01'],cwd=ROOT,check=True)
print('PASS: 120 family/English-Indic comparisons independently recomputed; 120 shuffle results; pure-family homes; exact expert capacities; 12 layer-position results; exact replica budgets; original mentor code/results unchanged.')
