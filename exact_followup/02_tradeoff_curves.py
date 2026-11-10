from common import *
rows=read('curves/table.csv') if (OUT/'curves/table.csv').exists() else [];checks=read('curves/feasibility.csv') if (OUT/'curves/feasibility.csv').exists() else [];meta={}
for x in rows:
 for field in ['gpus','R']:x[field]=int(x[field])
 for field in ['fit_seconds','fit_final_imbalance_max','local_share','remote_hit_reduction_pct','imbalance_mean','imbalance_max']:
  if x[field]!='':x[field]=float(x[field])
for x in checks:
 x['gpus']=int(x['gpus']);x['layer']=int(x['layer']);x['feasible']=x['feasible']=='True'
 if x['fit_primary_imbalance']!='':x['fit_primary_imbalance']=float(x['fit_primary_imbalance'])
for model,source in [('sarvam','wikipedia'),('qwen3','flores'),('sarvam','flores')]:
 data,langs,k=load(model,source);fc,ec=data['fit']['counts'],data['eval']['counts'];bylang=by_language(fc,data['fit']['language'],langs);name=f'{model}_{source}'
 for G in [4,8]:
  groups,unconstrained=groups_fit(bylang,G);fa=assignments(data['fit']['language'],langs,groups);ea=assignments(data['eval']['language'],langs,groups);fit=aggregate(fc,fa,G);ev=aggregate(ec,ea,G);meta[f'{name}_{G}']={'groups':[[langs[i] for i in z] for z in groups],'fit_tokens':int(data['fit']['tokens'].sum()),'eval_tokens':int(data['eval']['tokens'].sum())}
  if len([x for x in rows if x['dataset']==name and x['gpus']==G])==21:
   print('reused completed curve',name,G,flush=True);continue
  for cap in [1.1,1.25,1.5,2.,None]:
   where=[];start=time.perf_counter()
   for l in range(fit.shape[1]):
    prof=fit[:,l].T;loc=greedy(prof,prof.sum(1),cap);valid=loc is not None
    gpu_work=np.array([prof[loc==g].sum() for g in range(G)]) if valid else None
    checks.append({'dataset':name,'gpus':G,'cap':str(cap),'layer':l+1,'feasible':valid,'fit_primary_imbalance':float(gpu_work.max()/gpu_work.mean()) if valid else ''})
    if not valid:break
    assert all((loc==g).sum()==128//G for g in range(G));where.append(loc)
   secs=time.perf_counter()-start
   if len(where)!=fit.shape[1]:print('infeasible',name,G,cap,flush=True);continue
   where=np.array(where)
   for R in [0,2,4]:
    ex=copies(fit,where,R);z=evaluate(ev,where,ex);f=evaluate(fit,where,ex);rows.append({'dataset':name,'gpus':G,'R':R,'cap':str(cap),'method':'greedy','fit_seconds':secs,'fit_final_imbalance_max':f['imbalance_max'],**z})
   print('curve',name,G,cap,flush=True)
  for method in ['unconstrained_assignment','blind']:
   loc=unconstrained if method!='blind' else np.tile(np.arange(128)//(128//G),(fit.shape[1],1));f=fit if method!='blind' else np.repeat(fit.sum(0,keepdims=True)/G,G,axis=0);e=ev if method!='blind' else np.repeat(ev.sum(0,keepdims=True)/G,G,axis=0)
   for R in [0,2,4]:rows.append({'dataset':name,'gpus':G,'R':R,'cap':'reference','method':method,'fit_seconds':0,'fit_final_imbalance_max':'',**evaluate(e,loc,copies(f,loc,R,blind=method=='blind'))})
  write('curves/table.csv',rows);write('curves/feasibility.csv',checks)
(OUT/'curves/metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
plt=plot()
for R in [2,4]:
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for name,ax in zip(['sarvam_wikipedia','qwen3_flores','sarvam_flores'],axes):
  for G,color in [(4,'#147d86'),(8,'#b45309')]:
   z=[x for x in rows if x['dataset']==name and x['gpus']==G and x['R']==R and x['method']=='greedy'];ax.plot([x['imbalance_mean'] for x in z],[100*x['local_share'] for x in z],marker='o',label=f'{G} GPUs',color=color)
   for x in z:
    if x['cap'] in ['1.1','1.5']:ax.annotate(x['cap'],(x['imbalance_mean'],100*x['local_share']),xytext=(3,4),textcoords='offset points',fontsize=7)
  ax.set(title=name.replace('_',' '),xlabel='Mean held-out layer imbalance',ylabel='Local expert hits (%)');ax.grid(alpha=.3);ax.legend()
 fig.suptitle(f'Greedy locality–balance curves; R={R} copies/GPU/layer; fit-primary caps');fig.tight_layout();fig.savefig(OUT/f'curves/locality_imbalance_R{R}.png',dpi=160)
print('COMPLETE tradeoff curves',len(rows))
