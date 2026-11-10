from common import *
TRIALS=100;allrows=[];originals=[];metadata={}
for model,source in [('sarvam','wikipedia'),('qwen3','flores'),('sarvam','flores')]:
 for indic in [False,True]:
  name=f'{model}_{source}_{"indic" if indic else "full"}';records,langs,k=load(model,source,indic);fit=records['fit'];ev=records['eval'];G=4
  # Primary: real expert work per supplied request. Sensitivity: equal weight per sequence's routing distribution.
  for weighting in ['token_work','equal_sequence']:
   fc=fit['counts'] if weighting=='token_work' else fit['counts'].astype(float)/fit['tokens'][:,None,None]
   ec=ev['counts'] if weighting=='token_work' else ev['counts'].astype(float)/ev['tokens'][:,None,None]
   bylang=by_language(fc,fit['language'],langs);groups,where=groups_fit(bylang,G);fa=assignments(fit['language'],langs,groups);ea=assignments(ev['language'],langs,groups);fd=aggregate(fc,fa,G);ed=aggregate(ec,ea,G)
   default=np.tile(np.arange(128)//32,(where.shape[0],1));blindfit=np.repeat(fd.sum(0,keepdims=True)/G,G,axis=0);blindev=np.repeat(ed.sum(0,keepdims=True)/G,G,axis=0)
   for R in [2,4]:
    z=evaluate(ed,where,copies(fd,where,R));b=evaluate(blindev,default,copies(blindfit,default,R,blind=True));originals.append({'dataset':name,'weighting':weighting,'R':R,'condition':'original',**z});originals.append({'dataset':name,'weighting':weighting,'R':R,'condition':'blind_uniform_origins',**b})
   metadata[f'{name}_{weighting}']={'model':model,'source':source,'languages':langs,'groups':[[langs[i] for i in g] for g in groups],'fit_group_sizes':np.bincount(fa,minlength=G).tolist(),'eval_group_sizes':np.bincount(ea,minlength=G).tolist(),'fit_sequences':len(fc),'eval_sequences':len(ec),'fit_tokens':int(fit['tokens'].sum()),'eval_tokens':int(ev['tokens'].sum()),'layers':where.shape[0],'k':k,'fit_ids':fit['ids'].tolist(),'eval_ids':ev['ids'].tolist()}
   for blocked in ([False,True] if source=='flores' else [False]):
    control='parallel_sentence_blocked' if blocked else 'global_request_shuffle';trialseeds=np.random.SeedSequence(SEED).spawn(TRIALS);savedfit=[];savedeval=[]
    for trial,ss in enumerate(trialseeds):
     rngfit,rngev=[np.random.default_rng(s) for s in ss.spawn(2)];sfa=shuffled(fa,fit['parallel'],rngfit,blocked);sea=shuffled(ea,ev['parallel'],rngev,blocked)
     assert np.array_equal(np.bincount(fa,minlength=G),np.bincount(sfa,minlength=G));assert np.array_equal(np.bincount(ea,minlength=G),np.bincount(sea,minlength=G))
     sf=aggregate(fc,sfa,G);se=aggregate(ec,sea,G);sw=placement(sf)
     for R in [2,4]:
      z=evaluate(se,sw,copies(sf,sw,R));frozen=evaluate(se,where,copies(fd,where,R));allrows.append({'dataset':name,'weighting':weighting,'control':control,'trial':trial,'seed':SEED,'R':R,'frozen_plan_local_share':frozen['local_share'],**z})
     savedfit.append(sfa);savedeval.append(sea)
    folder=OUT/'shuffle';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/f'{name}_{weighting}_{control}_assignments.npz',fit=np.array(savedfit),eval=np.array(savedeval))
    write('shuffle/trials.csv',allrows);print('finished',name,weighting,control,flush=True)
write('shuffle/originals.csv',originals);write('shuffle/trials.csv',allrows);(OUT/'shuffle/metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
summary=[]
for original in originals:
 if original['condition']!='original':continue
 for control in ['global_request_shuffle','parallel_sentence_blocked']:
  rows=[x for x in allrows if all(x[k]==original[k] for k in ['dataset','weighting','R']) and x['control']==control]
  if not rows:continue
  vals=np.array([x['remote_hit_reduction_pct'] for x in rows]);base=next(x['remote_hit_reduction_pct'] for x in originals if x['dataset']==original['dataset'] and x['weighting']==original['weighting'] and x['R']==original['R'] and x['condition']=='blind_uniform_origins');summary.append({'dataset':original['dataset'],'weighting':original['weighting'],'R':original['R'],'control':control,'draws':len(vals),'original_reduction_pct':original['remote_hit_reduction_pct'],'shuffle_mean_reduction_pct':float(vals.mean()),'shuffle_p025':float(np.quantile(vals,.025)),'shuffle_p975':float(np.quantile(vals,.975)),'original_minus_shuffle_mean_pp':float(original['remote_hit_reduction_pct']-vals.mean()),'blind_reduction_pct':base,'original_advantage_over_blind_pp':original['remote_hit_reduction_pct']-base,'shuffle_advantage_over_blind_pp':float(vals.mean()-base),'draws_at_least_original':int(np.sum(vals>=original['remote_hit_reduction_pct']))})
write('shuffle/summary.csv',summary);print('COMPLETE exact shuffle',len(allrows),'evaluations')
