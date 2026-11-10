from pathlib import Path
import sys,json,csv,random,time,hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raj-exact-followup/2026-11-09';DATA=ROOT/'data/per_sequence';SEED=91026
sys.path.append(str(ROOT));sys.path.append(str(ROOT/'ablations'))
import analyze as core
from b_balance import greedy

def write(name,rows):
 f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True)
 with f.open('w',newline='') as o:
  w=csv.DictWriter(o,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def read(name):return list(csv.DictReader((OUT/name).open()))
def plot():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;return plt

def load(model,source,indic=False):
 d=np.load(DATA/f'{model}_per_sequence.npz');mask=(d['source']==source)&np.isin(d['split'],['fit','eval'])
 if indic:mask &= d['language']!='en'
 ix=np.flatnonzero(mask);assert len(ix);counts=d['counts'][ix];tokens=d['n_tokens'][ix];labels=d['language'][ix];split=d['split'][ix];ids=d['seq_id'][ix];parallel=d['flores_parallel_index'][ix]
 assert len(set(ids))==len(ids);assert np.all(tokens>0);k=8 if model=='qwen3' else 6
 assert np.all(counts.sum(-1,dtype=np.uint64)==k*tokens[:,None]);assert set(split)=={'fit','eval'}
 langs=sorted(set(labels));expected=9 if indic else 10;assert len(langs)==expected
 return {s:{'counts':counts[split==s],'tokens':tokens[split==s],'language':labels[split==s],'ids':ids[split==s],'parallel':parallel[split==s]} for s in ['fit','eval']},langs,k

def by_language(c,labels,langs):return np.array([c[labels==l].sum(0,dtype=np.float64) for l in langs])
def normalize(c):return c/c.sum((0,2),keepdims=True)
def groups_fit(bylang,G,samples=300):
 p=normalize(bylang);best=-1;chosen=None;bestwhere=None;rng=random.Random(SEED)
 for groups in core.balanced_groupings(len(p),G,samples,rng):
  demand=np.array([p[g].sum(0) for g in groups]);where=placement(demand);score=local_objective(demand,where)
  if score>best+1e-12:best=score;chosen=groups;bestwhere=where
 return chosen,bestwhere

def placement(demand,cap=None):
 G,L,E=demand.shape;where=[]
 for l in range(L):
  prof=demand[:,l].T
  if cap is None:
   rows,cols=linear_sum_assignment(-np.repeat(prof,E//G,axis=1));a=np.empty(E,int);a[rows]=cols//(E//G)
  else:a=greedy(prof,prof.sum(1),cap)
  assert a is not None;where.append(a)
 return np.array(where)
def local_objective(demand,where):return sum(demand[g,l,where[l]==g].sum() for g in range(len(demand)) for l in range(where.shape[0]))
def copies(demand,where,R,blind=False):
 G,L,E=demand.shape;extra=np.zeros((L,G,E),bool)
 for l in range(L):
  total=demand[:,l].sum(0)
  for g in range(G):
   candidates=np.flatnonzero(where[l]!=g);values=total[candidates] if blind else demand[g,l,candidates];extra[l,g,candidates[np.argsort(-values,kind='stable')[:R]]]=True
 return extra

def evaluate(demand,where,extra):
 G,L,E=demand.shape;local=[];loads=[]
 for l in range(L):
  loc=0.;work=np.zeros(G)
  for g in range(G):
   here=(where[l]==g)|extra[l,g];loc+=demand[g,l,here].sum();work+=np.bincount(np.where(here,g,where[l]),weights=demand[g,l],minlength=G)
  local.append(loc);loads.append(work)
 loads=np.array(loads);assert np.allclose(loads.sum(1),1);imb=loads.max(1)/loads.mean(1);loc=float(np.mean(local))
 return {'local_share':loc,'remote_hit_reduction_pct':100*(loc-1/G)/(1-1/G),'imbalance_mean':float(imb.mean()),'imbalance_max':float(imb.max())}

def aggregate(c,assignment,G):
 a=np.array([c[assignment==g].sum(0,dtype=np.float64) for g in range(G)]);return normalize(a)
def assignments(labels,langs,groups):
 lookup={langs[l]:g for g,group in enumerate(groups) for l in group};return np.array([lookup[l] for l in labels],dtype=np.uint8)
def shuffled(original,parallel,rng,blocked=False):
 if not blocked:return rng.permutation(original)
 out=original.copy()
 for idx in np.unique(parallel):
  pos=np.flatnonzero(parallel==idx);out[pos]=rng.permutation(original[pos])
 return out
