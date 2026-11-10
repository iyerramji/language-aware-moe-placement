"""Independent accounting checks plus parity with mentor's simulation."""
from common import *
import itertools
p,q,langs,m=data();w=np.ones(len(langs))/len(langs)
for g in [4,8]:
 groups=baseline_groups(g,langs);where=layout(p,w,groups)
 for R in [0,2,4]:
  ours=evaluate(q,w,groups,where,copies(p,w,groups,where,R));ref=core.simulate(p,q,groups,w,replicas=R)
  assert np.allclose(ours['per_layer_local'],ref['local_rep'],atol=1e-12)
  assert np.allclose(ours['per_layer_imbalance'],ref['imb_rep'],atol=1e-12)
  assert np.allclose(ours['per_layer_load'].sum(-1),1)
 dflt=default_plan(p,g);no=np.zeros((p.shape[1],g,p.shape[2]),bool);met=evaluate(q,w,groups,dflt,no,blind=True);assert abs(met['local_share']-1/g)<1e-12
# Hand-check one layer, 2 GPUs, 2 languages, unequal traffic, with and without a remote copy.
toy=np.array([[[.7,.3]],[[.2,.8]]]);groups=[[0],[1]];where=np.array([[0,1]]);weights=np.array([.25,.75]);extra=np.zeros((1,2,2),bool)
z=evaluate(toy,weights,groups,where,extra);assert abs(z['local_share']-(.25*.7+.75*.8))<1e-12
extra[0,0,1]=True;z=evaluate(toy,weights,groups,where,extra);assert abs(z['local_share']-(.25+.75*.8))<1e-12;assert np.allclose(z['per_layer_load'],[[.25+.75*.2,.75*.8]])
# Check ILP objective against exhaustive feasible assignments on a tiny instance.
from b_balance import solve
prof=np.array([[.3,.1],[.2,.3],[.1,.2],[.4,.4]]);load=np.array([.1,.2,.3,.4]);cap=1.2
sol,secs,opt,status=solve(prof,load,cap);scores=[]
for left in itertools.combinations(range(4),2):
 a=np.ones(4,int);a[list(left)]=0
 if max(load[a==g].sum() for g in range(2))<=cap*load.sum()/2+1e-9:scores.append(prof[np.arange(4),a].sum())
assert opt;assert abs(prof[np.arange(4),sol].sum()-max(scores))<1e-9
summary=json.loads((ROOT/'results/summary.json').read_text());assert abs(summary['ep']['4']['local_share_aware']-.344)<=.001;assert abs(summary['ep']['4']['local_share_baseline']-.257)<=.001
print('PASS: baseline targets, all-layer mentor parity, independent replica/load arithmetic, and tiny exhaustive ILP comparison.')
import csv
rows={}
for folder,n in [('A',36),('B',30),('C',10),('D',72)]:
 with (OUT/folder/'table.csv').open() as f:rows[folder]=list(csv.DictReader(f))
 assert len(rows[folder])==n
 for row in rows[folder]:
  for field in ['local_share','traffic_cut_pct','imbalance_mean','imbalance_max']:assert np.isfinite(float(row[field]))
  assert 0<=float(row['local_share'])<=1;assert float(row['imbalance_mean'])>=1-1e-9
for row in rows['D']:
 if row['fit_mix']==row['serve_mix']=='equal':
  ref=next(x for x in rows['A'] if x['gpus']=='4' and x['R']==row['R'] and x['policy']=='aware');assert abs(float(ref['traffic_cut_pct'])-float(row['traffic_cut_pct']))<1e-9
for alloc in json.loads((OUT/'C/allocations.json').read_text()).values():assert np.all(np.array(alloc).sum(0)==96)
proofs=json.loads((OUT/'B/solver-details.json').read_text());assert len(proofs)==240;assert all(x['proven_optimal'] and x['feasible_incumbent'] for x in proofs)
print('PASS: complete A/B/C/D grid counts; all metrics finite; D equal-mix parity with A; exact C budgets; 240/240 four-GPU IP proofs.')
