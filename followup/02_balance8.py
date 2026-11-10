from common import *
from b_balance import greedy
p,q,langs,meta=data();w=np.ones(len(langs))/len(langs);rows=[];details=[];saved={}
for G in [4,8]:
 groups=baseline_groups(G,langs)
 for cap in [1.1,1.25,1.5,2.,None]:
  loc=[];elapsed=0
  for l in range(p.shape[1]):
   prof=core.profits(p,groups,w,l);load=(w[:,None]*p[:,l]).sum(0);st=time.perf_counter();where=greedy(prof,load,cap);secs=time.perf_counter()-st;elapsed+=secs
   feasible=where is not None;imb=float(max(load[where==g].sum() for g in range(G))/(load.sum()/G)) if feasible else None
   details.append({'gpus':G,'cap':str(cap),'layer':l+1,'feasible':feasible,'fit_primary_imbalance':imb,'seconds':secs})
   if not feasible:break
   assert all((where==g).sum()==128//G for g in range(G));assert cap is None or imb<=cap+1e-6;loc.append(where)
  if len(loc)!=p.shape[1]:continue
  loc=np.array(loc);saved[f'{G}_{cap}']=loc
  for R in [0,2,4]:
   ex=copies(p,w,groups,loc,R);met=evaluate(q,w,groups,loc,ex);fit=evaluate(p,w,groups,loc,ex);rows.append({'gpus':G,'cap':str(cap),'R':R,'method':'greedy','fit_seconds':elapsed,'fit_primary_imbalance_max':max(x['fit_primary_imbalance'] for x in details if x['gpus']==G and x['cap']==str(cap)),'fit_final_imbalance_max':fit['imbalance_max'],**metric(met)})
  write_csv('balance8/table.csv',rows);print('balance done',G,cap,flush=True)
 for method in ['blind','unconstrained_assignment']:
  loc=default_plan(p,G) if method=='blind' else layout(p,w,groups)
  for R in [0,2,4]:
   ex=copies(p,w,groups,loc,R,blind=method=='blind');met=evaluate(q,w,groups,loc,ex,blind=method=='blind');fit=evaluate(p,w,groups,loc,ex,blind=method=='blind');rows.append({'gpus':G,'cap':'reference','R':R,'method':method,'fit_seconds':0,'fit_primary_imbalance_max':'','fit_final_imbalance_max':fit['imbalance_max'],**metric(met)})
write_csv('balance8/table.csv',rows);write_csv('balance8/layer-checks.csv',details);np.savez(OUT/'balance8/placements.npz',**saved)
from render_figures import render_balance
render_balance()
notes=['# B on eight GPUs, greedy implementation','All48layers, caps1.1/1.25/1.5/2/none, R0/2/4. Four-GPU same-method rerun provides a paired reference. Greedy is not a certified global optimum. Original B semantics retained: cap constrains FIT PRIMARY expert load before adding replicas. Held-out final expert load can exceed the cap. Every GPU has E/G primaries and exactly R extra experts. Groups fixed to original baseline. No GPU/cloud execution.']
for G in [4,8]:
 for cap in ['1.1','1.25']:
  x=next(z for z in rows if z['gpus']==G and z['cap']==cap and z['R']==2);notes.append(f"- G{G},cap{cap},R2: reduction {x['remote_hit_reduction_pct']:.3f}%, mean imbalance {x['imbalance_mean']:.3f}, worst layer {x['imbalance_max']:.3f}; fit time {x['fit_seconds']:.2f}s.")
(OUT/'balance8/NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
