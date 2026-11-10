from common import *
p,q,langs,meta=data();rows=[];draws=[];plans={};seed=20261109;rng=np.random.default_rng(seed)
for name in ['indic_only','equal']:
 groups,w,where=get_plan(name);G=4;active=np.flatnonzero(w>0);pbar=(p*w[:,None,None]).sum(0);qbar=(q*w[:,None,None]).sum(0);pmix=np.repeat(pbar[None],len(w),axis=0);qmix=np.repeat(qbar[None],len(w),axis=0)
 nullwhere=layout(pmix,w,groups)
 for R in [2,4]:
  ex=copies(p,w,groups,where,R);nex=copies(pmix,w,groups,nullwhere,R);default=default_plan(p,G);bex=copies(p,w,groups,default,R,blind=True)
  conditions=[('original',q,where,ex,False),('request_shuffle_expectation_frozen_plan',qmix,where,ex,False),('request_shuffle_expectation_refitted_plan',qmix,nullwhere,nex,False),('language_blind_uniform_origin',q,default,bex,True)]
  for condition,ev,loc,extra,blind in conditions:rows.append({'mix':name,'gpus':G,'R':R,'condition':condition,**metric(evaluate(ev,w,groups,loc,extra,blind=blind))})
  plans[f'{name}_R{R}']={'groups':groups,'weights':w.tolist(),'original_where':where.tolist(),'original_extra':ex.tolist(),'null_where':nullwhere.tolist(),'null_extra':nex.tolist()}
 # Secondary control: move whole language profiles, NOT requests. Group sizes preserved, label-conditioned info is retained.
 seen=set()
 for i in range(100):
  perm=rng.permutation(active)
  # Assign old group slots to new identities; full profile permutation between train/eval together.
  mapping=dict(zip(active,perm));shuffled=[[int(mapping.get(l,l)) for l in group] for group in groups];loc=layout(p,w,shuffled)
  for R in [2,4]:draws.append({'mix':name,'draw':i,'R':R,'groups':json.dumps(shuffled,separators=(',',':')),**metric(evaluate(q,w,shuffled,loc,copies(p,w,shuffled,loc,R)))})
 print('shuffle done',name,flush=True)
write_csv('shuffle/table.csv',rows);write_csv('shuffle/whole-language-shuffles.csv',draws);(OUT/'shuffle/plans.json').write_text(json.dumps(plans,separators=(',',':'))+'\n')
plt=plot();fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,name in zip(axes,['indic_only','equal']):
 z=[x for x in rows if x['mix']==name and x['R']==2];ax.bar(range(len(z)),[x['remote_hit_reduction_pct'] for x in z]);ax.set_xticks(range(len(z)),['Original','Shuffle: fixed plan','Shuffle: refit','Blind, uniform origins'],rotation=20,ha='right');ax.set(title=name.replace('_',' '),ylabel='Remote expert-hit reduction (%)');ax.grid(axis='y',alpha=.3)
fig.suptitle('Expected request shuffle, preserved home-group masses; 4 GPUs, R2');fig.tight_layout();fig.savefig(OUT/'shuffle/comparison.png',dpi=160)
notes=['# Language-information controls','The supplied counts are aggregated by language/half, with 48 sequences of512tokens per fit/evaluation half and language. Request-level expert counts are absent. Therefore the exact random-label realization and finite-request shuffle variance cannot be reconstructed.','The expectation control replaces every pseudo-language profile by the same weighted aggregate profile, separately on fit and evaluation halves, preserving total expert demand and each original home group’s mass. This is the expected demand under independently randomized request-to-group assignments with fixed group sizes for these equal-length sequences, not an observed shuffled dataset. Report both frozen-plan and expectation-refitted-plan variants. Max-load metrics are computed on expected loads, not expected finite-shuffle maxima.','The separate100whole-language-profile permutations keep language packets intact and refit placement. This is a weaker random-grouping control, NOT request-label randomization and NOT evidence that language information is unnecessary. No permutation p-value is claimed for the expectation control.','No group is selected on evaluation data. Full mix shuffles include English; Indic-only excludes English requests. Blind comparator spreads every language uniformly across GPUs, whereas the random-homing expectation preserves original unequal home-group masses. This distinction can itself provide locality benefits unrelated to language.']
for name in ['indic_only','equal']:
 for R in [2,4]:
  d={x['condition']:x for x in rows if x['mix']==name and x['R']==R};orig=d['original']['remote_hit_reduction_pct'];ref=d['request_shuffle_expectation_refitted_plan']['remote_hit_reduction_pct'];blind=d['language_blind_uniform_origin']['remote_hit_reduction_pct'];notes.append(f'- {name},R{R}: original {orig:.3f}%, shuffled expectation/refit {ref:.3f}%, blind {blind:.3f}%; original-minus-null {orig-ref:.3f} percentage points. Do not translate directly to network bytes.')
(OUT/'shuffle/NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
