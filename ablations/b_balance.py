from common import *
import pulp
import argparse

def greedy(prof,load,cap):
 E,G=prof.shape;S=E//G;target=load.sum()/G;where=np.full(E,-1,int);counts=np.zeros(G,int);used=np.zeros(G)
 # Assign heavy experts first to least-loaded available GPU; deterministic.
 for e in np.argsort(-load,kind='stable'):
  cand=np.flatnonzero(counts<S);g=min(cand,key=lambda g:(used[g],-prof[e,g],g));where[e]=g;counts[g]+=1;used[g]+=load[e]
 if cap is not None and used.max()>cap*target+1e-8:return None
 # Best feasible positive-gain pair swaps preserve capacity and expert count.
 for _ in range(256):
  best=None;gain=1e-12
  for e in range(E):
   g=where[e]
   for f in range(e+1,E):
    h=where[f]
    if g==h:continue
    delta=prof[e,h]+prof[f,g]-prof[e,g]-prof[f,h]
    if delta<=gain:continue
    if cap is not None and max(used[g]-load[e]+load[f],used[h]-load[f]+load[e])>cap*target+1e-8:continue
    gain=delta;best=(e,f,g,h)
  if best is None:break
  e,f,g,h=best;where[e],where[f]=h,g;used[g]+=load[f]-load[e];used[h]+=load[e]-load[f]
 return where

def solve(prof,load,cap):
 E,G=prof.shape;S=E//G;model=pulp.LpProblem('placement',pulp.LpMaximize);x=pulp.LpVariable.dicts('x',(range(E),range(G)),cat='Binary')
 model+=pulp.lpSum(float(prof[e,g])*x[e][g] for e in range(E) for g in range(G))
 for e in range(E):model+=pulp.lpSum(x[e][g] for g in range(G))==1
 for g in range(G):
  model+=pulp.lpSum(x[e][g] for e in range(E))==S
  if cap is not None:model+=pulp.lpSum(float(load[e])*x[e][g] for e in range(E))<=cap*float(load.sum())/G
 start=time.perf_counter();model.solve(pulp.PULP_CBC_CMD(msg=False,timeLimit=8,threads=1,options=['randomSeed 0']));elapsed=time.perf_counter()-start
 status=pulp.LpStatus[model.status];optimal=model.sol_status==pulp.LpSolutionOptimal
 where=np.array([max(range(G),key=lambda g:x[e][g].value() or 0) for e in range(E)])
 valid=all((where==g).sum()==S for g in range(G)) and (cap is None or max(load[where==g].sum() for g in range(G))<=cap*load.sum()/G+1e-7)
 return where if valid else None,elapsed,optimal,status

def main():
 args=argparse.ArgumentParser();args.add_argument('--gpus',type=int,default=4);g=args.parse_args().gpus
 p,q,langs,m=data();w=np.ones(len(langs))/len(langs);groups=baseline_groups(g,langs);folder=OUT/('B' if g==4 else 'B8');folder.mkdir(parents=True,exist_ok=True);rows=[];details=[]
 for cap in [1.1,1.25,1.5,2.0,None]:
  plans={'integer_program':[],'greedy':[]};times={'integer_program':0.,'greedy':0.};opts=0;valids=0
  for l in range(p.shape[1]):
   prof=core.profits(p,groups,w,l);load=(w[:,None]*p[:,l]).sum(0);exact,secs,opt,status=solve(prof,load,cap);times['integer_program']+=secs;opts+=int(opt);valids+=int(exact is not None)
   st=time.perf_counter();gr=greedy(prof,load,cap);gs=time.perf_counter()-st;times['greedy']+=gs
   details.append({'cap':cap,'layer':l+1,'status':status,'proven_optimal':opt,'feasible_incumbent':exact is not None,'solve_seconds':secs,'greedy_seconds':gs,'ip_fit_objective':float(prof[np.arange(p.shape[2]),exact].sum()) if exact is not None else None,'greedy_fit_objective':float(prof[np.arange(p.shape[2]),gr].sum()) if gr is not None else None})
   plans['integer_program'].append(exact);plans['greedy'].append(gr)
  for method,layers in plans.items():
   if any(x is None for x in layers):continue
   where=np.array(layers);obj=np.mean([core.profits(p,groups,w,l)[np.arange(p.shape[2]),where[l]].sum() for l in range(p.shape[1])])
   for R in [0,2,4]:
    met=evaluate(q,w,groups,where,copies(p,w,groups,where,R));rows.append({'gpus':g,'cap':str(cap),'R':R,'method':method,'fit_objective':float(obj),'solve_seconds':times[method],'optimal_layers':opts if method=='integer_program' else 0,'layers':p.shape[1],**record(met)})
  (folder/'solver-details.json').write_text(json.dumps(details,indent=2)+'\n');save_table(folder,rows);print('finished cap',cap,'optimal',opts,'feasible',valids,flush=True)
 np.savez(folder/'accounting.npz',fit_weights=w)
 plt=plot();fig,ax=plt.subplots(figsize=(8,5))
 for method in ['integer_program','greedy']:
  for R in [0,2,4]:
   z=[x for x in rows if x['method']==method and x['R']==R];ax.plot([x['imbalance_mean'] for x in z],[x['traffic_cut_pct'] for x in z],marker='o',label=f'{method}, R={R}')
 for R in [0,2,4]:
  where=default_plan(p,g);met=evaluate(q,w,groups,where,copies(p,w,groups,where,R,blind=True),blind=True);ax.scatter(met['imbalance_mean'],met['traffic_cut_pct'],marker='x',color='black');ax.annotate(f'blind R={R}',(met['imbalance_mean'],met['traffic_cut_pct']),fontsize=7)
 ax.set(xlabel='Mean per-layer busiest GPU / mean (held-out)',ylabel='Remote expert-hit reduction (%)',title='B: locality and load balance');ax.legend(fontsize=7);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(folder/'frontier.png',dpi=160)
 notes=['# B: balance-aware placement',f'{g} simulated GPUs. All 48 layers solved per cap; solver proof flags and incumbents are recorded separately. Cap constrains fit-half primary expert load BEFORE copies; it is not a guarantee on held-out or replica-adjusted load. Greedy is a deterministic heavy-first assignment plus feasible positive pair swaps; it is not globally optimal. An IP is only called exact when all 48 layers have optimal solution certificates.']
 for x in rows:
  if x['method']=='integer_program' and x['R']==2:notes.append(f"- Cap {x['cap']}: traffic cut {x['traffic_cut_pct']:.3f}%, held-out mean imbalance {x['imbalance_mean']:.3f}, optimal layers {x['optimal_layers']}/48, total solve time {x['solve_seconds']:.2f}s.")
 (folder/'NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
if __name__=='__main__':main()
