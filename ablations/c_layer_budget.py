from common import *
def specificity_budget(spec,total,L,G,max_r):
 raw=spec/spec.sum()*total;n=np.minimum(np.floor(raw).astype(int),max_r)
 while n.sum()<total:
  candidates=np.flatnonzero(n<max_r);i=max(candidates,key=lambda i:(raw[i]-n[i],-i));n[i]+=1
 return np.repeat(n[:,None],G,axis=1)
def main():
 p,q,langs,m=data();w=np.ones(len(langs))/len(langs);L=p.shape[1];total=2*L;spec=core.js_bits(p,p.mean(0)[None]).mean(0);rows=[];folder=OUT/'C';folder.mkdir(parents=True,exist_ok=True);plt=plot();fig,axes=plt.subplots(2,1,figsize=(11,7));allocs={}
 for g,ax in zip([4,8],axes):
  groups=baseline_groups(g,langs);where=layout(p,w,groups);budgets={'uniform':np.full((L,g),2),'specificity':specificity_budget(spec,total,L,g,p.shape[2]-p.shape[2]//g)}
  # Since copies do not compete across GPUs, top 96 marginal expert-hit gains per GPU are globally optimal for this fixed layout/objective.
  n=np.zeros((L,g),int)
  for gpu in range(g):
   gains=[]
   for l in range(L):
    prof=core.profits(p,groups,w,l);cand=np.flatnonzero(where[l]!=gpu);values=sorted(prof[cand,gpu],reverse=True)
    gains.extend((float(v),l,k) for k,v in enumerate(values))
   for v,l,k in sorted(gains,key=lambda x:(-x[0],x[1],x[2]))[:total]:n[l,gpu]+=1
  budgets['greedy_gain']=n
  for k in [12,24]:
   n=np.zeros((L,g),int);selected=np.argsort(-spec,kind='stable')[:k];n[selected]=total//k;budgets[f'concentrated_{k}']=n
  for method,b in budgets.items():
   assert np.all(b.sum(0)==total);ex=copies(p,w,groups,where,b);assert np.all(ex.sum((0,2))==total);met=evaluate(q,w,groups,where,ex);rows.append({'gpus':g,'method':method,'copies_per_gpu_total':total,**record(met)});ax.plot(np.arange(1,L+1),100*(met['per_layer_local']-1/g)/(1-1/g),label=method);allocs[f'{g}_{method}']=b.tolist()
  ax.set(title=f'{g} simulated GPUs; 96 copies per GPU',xlabel='Layer',ylabel='Remote expert-hit reduction (%)');ax.legend(fontsize=7);ax.grid(alpha=.3)
 fig.tight_layout();fig.savefig(folder/'per_layer.png',dpi=160);save_table(folder,rows);(folder/'allocations.json').write_text(json.dumps(allocs,indent=2)+'\n')
 notes=['# C: fixed layer budget','Executed fixed-layout CPU simulation, 96 copies per GPU. Specificity measured on fit half; largest-deficit rounding preserves exact budgets. Greedy uses fit-half marginal local-hit gains, with no held-out tuning. Equal-sized experts are assumed.']
 for g in [4,8]:
  z=[x for x in rows if x['gpus']==g];best=max(z,key=lambda x:x['traffic_cut_pct']);uniform=next(x for x in z if x['method']=='uniform');notes.append(f"- G={g}: best held-out policy {best['method']} at {best['traffic_cut_pct']:.3f}% traffic cut, {best['traffic_cut_pct']-uniform['traffic_cut_pct']:.3f} points over uniform; imbalance {best['imbalance_mean']:.3f}. Winner is descriptive, not selected using evaluation data for deployment.")
 (folder/'NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
if __name__=='__main__':main()
