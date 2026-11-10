from common import *
rows=read('curves/table.csv');rows=[x for x in rows if x['dataset']!='qwen3_wikipedia_aggregate'];prior=ROOT/'results/raj-followup/2026-11-09/balance8/table.csv'
for x in csv.DictReader(prior.open()):
 d={k:x.get(k,'') for k in rows[0]};d['dataset']='qwen3_wikipedia_aggregate';rows.append(d)
write('curves/table.csv',rows)
(OUT/'curves/prior-source.json').write_text(json.dumps({'dataset':'qwen3_wikipedia_aggregate','source':str(prior.relative_to(ROOT)),'sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'weighting':'Earlier equal-language-token weighting; new per-sequence curves use actual input-token work. Kept in a separate panel.','rerun':False},indent=2)+'\n')
bounds=[];impossible=[]
for model,source in [('sarvam','wikipedia'),('qwen3','flores'),('sarvam','flores')]:
 records,langs,k=load(model,source);expert=records['fit']['counts'].sum(0,dtype=np.float64)
 for G in [4,8]:
  for l,work in enumerate(expert):
   hot=int(work.argmax());others=np.sort(np.delete(work,hot));bound=G*(work[hot]+others[:128//G-1].sum())/work.sum();bounds.append({'dataset':f'{model}_{source}','gpus':G,'layer':l+1,'hottest_expert':hot,'primary_slot_lower_bound':float(bound)})
  for cap in [1.1,1.25,1.5,2.]:
   peak=max((x for x in bounds if x['dataset']==f'{model}_{source}' and x['gpus']==G),key=lambda x:x['primary_slot_lower_bound'])
   if peak['primary_slot_lower_bound']>cap+1e-8:impossible.append({'dataset':f'{model}_{source}','gpus':G,'cap':cap,'witness_layer':peak['layer'],'minimum_primary_imbalance_bound':peak['primary_slot_lower_bound']})
write('curves/fit_lower_bounds.csv',bounds);write('curves/infeasible_caps.csv',impossible)
plt=plot()
for R in [2,4]:
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for name,ax in zip(['qwen3_wikipedia_aggregate','sarvam_wikipedia','qwen3_flores','sarvam_flores'],axes.flat):
  for G,color in [(4,'#147d86'),(8,'#b45309')]:
   z=[x for x in rows if x['dataset']==name and x['gpus']==str(G) and x['R']==str(R) and x['method']=='greedy'];ax.plot([float(x['imbalance_mean']) for x in z],[100*float(x['local_share']) for x in z],marker='o',label=f'{G} GPUs',color=color)
   for x in z:
    if x['cap'] in ['1.1','1.5']:ax.annotate(x['cap'],(float(x['imbalance_mean']),100*float(x['local_share'])),xytext=(3,4),textcoords='offset points',fontsize=7)
  title=name.replace('_',' ');ax.set(title=title,xlabel='Mean held-out layer imbalance',ylabel='Local expert hits (%)');ax.grid(alpha=.3);ax.legend()
 fig.suptitle(f'Locality–balance curves; R={R}; cap on primary fit load, before copies');fig.tight_layout();fig.savefig(OUT/f'curves/locality_imbalance_R{R}.png',dpi=160)
print('Added separate prior Qwen3 Wikipedia panel; total curve rows',len(rows))
