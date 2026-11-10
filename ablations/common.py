"""Shared held-out simulation accounting; no GPU execution."""
import sys,json,csv,time,random,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze as core
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/ablations'
def data():
 d=np.load(ROOT/'results/routing_qwen3_30b_a3b.npz');c=d['counts'].astype(float);m=json.loads(str(d['meta']));return c[0]/c[0].sum(-1,keepdims=True),c[1]/c[1].sum(-1,keepdims=True),list(m['langs']),m

def baseline_groups(g,langs):
 s=json.loads((ROOT/'results/summary.json').read_text());return [[langs.index(l) for l in group] for group in s['ep'][str(g)]['groups']]
def choose_groups(p,w,g):
 rng=random.Random(0);best=None;score=-1
 for groups in core.balanced_groupings(len(w),g,300,rng):
  z=sum(core.profits(p,groups,w,l)[np.arange(p.shape[2]),core.place_layer(core.profits(p,groups,w,l),p.shape[2]//g)].sum() for l in range(p.shape[1]))
  if z>score:score=z;best=groups
 return best

def layout(p,w,groups):return np.array([core.place_layer(core.profits(p,groups,w,l),p.shape[2]//len(groups)) for l in range(p.shape[1])])
def copies(p,w,groups,where,budget,blind=False):
 L,E=where.shape;G=len(groups);extra=np.zeros((L,G,E),bool)
 for l in range(L):
  prof=core.profits(p,groups,w,l);global_pop=(w[:,None]*p[:,l]).sum(0)
  for g in range(G):
   cand=np.flatnonzero(where[l]!=g);order=np.argsort(-(global_pop[cand] if blind else prof[cand,g]),kind='stable');n=int(budget[l,g]) if np.ndim(budget) else int(budget);extra[l,g,cand[order[:n]]]=True
 return extra

def evaluate(p,w,groups,where,extra,blind=False):
 N,L,E=p.shape;G=len(groups);local=[];imb=[];per=[];home=np.empty(N,int)
 for g,ls in enumerate(groups):home[ls]=g
 for l in range(L):
  loads=np.zeros(G);loc=0
  for origin in range(G):
   demand=(w[:,None]*p[:,l]).sum(0)/G if blind else (w[home==origin,None]*p[home==origin,l]).sum(0)
   mask=(where[l]==origin)|extra[l,origin];loc+=demand[mask].sum();served=np.where(mask,origin,where[l]);loads+=np.bincount(served,weights=demand,minlength=G)
  local.append(loc);imb.append(loads.max()/loads.mean());per.append(loads.tolist())
 local=np.array(local);base=1/G
 return {'local_share':float(local.mean()),'traffic_cut_pct':float(100*(local.mean()-base)/(1-base)),'imbalance_mean':float(np.mean(imb)),'imbalance_max':float(np.max(imb)),'per_layer_local':local,'per_layer_imbalance':np.array(imb),'per_layer_load':np.array(per)}
def default_plan(p,g):return np.tile(np.arange(p.shape[2])//(p.shape[2]//g),(p.shape[1],1))
def record(metrics):return {k:v for k,v in metrics.items() if not k.startswith('per_layer')}
def save_table(folder,rows):
 folder.mkdir(parents=True,exist_ok=True)
 with (folder/'table.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def plot():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;return plt
