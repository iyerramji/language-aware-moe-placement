from common import *
p,q,langs,meta=data();G=4;old=json.loads((ROOT/'results/organized-experiments/raj-requested/D/plans.json').read_text());candidates=[[[langs.index(l) for l in group] for group in x['groups']] for x in old.values()]
anchors=[core.weights(langs,s) for s in ['', 'hi=0.5,en=0.3','en=0.6','en=0']];trajectories={'gradual':np.array([(1-a)*anchors[i]+a*anchors[(i+1)%4] for i in range(4) for a in np.arange(8)/8]),'abrupt':np.repeat(np.array(anchors),8,axis=0),'alternating':np.array([anchors[i%2] for i in range(32)])}
cache={};fitmeta={}
def key(w):return tuple(np.round(w,12))
def plan(w):
 k=key(w)
 if k not in cache:
  best=-1;selected=None;where=None;st=time.perf_counter()
  for groups in candidates:
   loc=layout(p,w,groups);score=sum(core.profits(p,groups,w,l)[np.arange(128),loc[l]].sum() for l in range(48))
   if score>best+1e-12:best=score;selected=groups;where=loc
  cache[k]=(selected,where,{R:copies(p,w,selected,where,R) for R in [2,4]});fitmeta[str(k)]={'weights':w.tolist(),'groups':selected,'fit_seconds':time.perf_counter()-st}
 return cache[k]
for name,ws in trajectories.items():
 for w in ws:plan(w)
 print('planned',name,'distinct cached',len(cache),flush=True)
rows=[];blocks=[]
for name,ws in trajectories.items():
 for R in [2,4]:
  for period in [1,2,4,8,16,32]:
   for mode in ['current_mix_oracle','previous_block_mix']:
    current=None;records=[];rebuilds=0
    for t,w in enumerate(ws):
     if t%period==0:
      fitw=w if mode=='current_mix_oracle' else ws[max(0,t-1)];current=plan(fitw);rebuilds+=1
     group,loc,extras=current;freshg,freshloc,freshex=plan(w);z=evaluate(q,w,group,loc,extras[R]);fresh=evaluate(q,w,freshg,freshloc,freshex[R]);initial=plan(ws[0]);fixed=evaluate(q,w,initial[0],initial[1],initial[2][R]);records.append((z,fresh,fixed))
     blocks.append({'trajectory':name,'R':R,'period_blocks':period,'information':mode,'block':t,'remote_hit_reduction_pct':z['traffic_cut_pct'],'fresh_reduction_pct':fresh['traffic_cut_pct'],'fixed_reduction_pct':fixed['traffic_cut_pct'],'mean_imbalance':z['imbalance_mean']})
    stale=np.mean([x[0]['traffic_cut_pct'] for x in records]);fresh=np.mean([x[1]['traffic_cut_pct'] for x in records]);fixed=np.mean([x[2]['traffic_cut_pct'] for x in records]);retention=stale/fresh if fresh>0 else float('nan');improvement=fresh-fixed;recovered=(stale-fixed)/improvement if improvement>1e-9 else float('nan')
    rows.append({'trajectory':name,'R':R,'period_blocks':period,'information':mode,'blocks':len(ws),'rebuild_count':rebuilds,'reduction_pct':float(stale),'fresh_reduction_pct':float(fresh),'fixed_reduction_pct':float(fixed),'fraction_fresh_saving':float(retention),'fraction_adaptation_gain_recovered':float(recovered),'mean_imbalance':float(np.mean([x[0]['imbalance_mean'] for x in records]))})
 write_csv('rebuild/table.csv',rows);print('evaluated',name,flush=True)
write_csv('rebuild/table.csv',rows);write_csv('rebuild/block-results.csv',blocks);(OUT/'rebuild/trajectories.json').write_text(json.dumps({n:[dict(zip(langs,w.tolist())) for w in ws] for n,ws in trajectories.items()},indent=2)+'\n');(OUT/'rebuild/fit-cache.json').write_text(json.dumps(fitmeta,indent=2)+'\n')
plt=plot();fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
for name,ax in zip(trajectories,axes):
 for mode in ['current_mix_oracle','previous_block_mix']:
  z=[x for x in rows if x['trajectory']==name and x['R']==2 and x['information']==mode];ax.plot([x['period_blocks'] for x in z],[100*x['fraction_fresh_saving'] for x in z],marker='o',label=mode)
 ax.axhline(90,color='grey',linestyle='--');ax.set(title=name,xlabel='Blocks between rebuilds');ax.set_xscale('log',base=2);ax.grid(alpha=.3)
axes[0].set_ylabel('Fresh-plan saving retained (%)');axes[0].legend(fontsize=7);fig.suptitle('Synthetic traffic trajectories, 32 blocks, 4 GPUs, R2');fig.tight_layout();fig.savefig(OUT/'rebuild/frequency.png',dpi=160)
notes=['# Rebuild frequency under synthetic mixture changes','Three PREDEFINED synthetic32-block trajectories: gradual transitions among equal/Hindi-heavy/English-heavy/Indic-only; abrupt8-block regimes; alternating equal/Hindi-heavy each block. These are not recorded production arrivals or observed temporal routing drift. Each block has equal total work. Within-language fit/evaluation profiles remain fixed.','The bounded planner searches only six pre-existing candidate home groupings from D, choosing on fit-half primary locality and then refitting layout/copies. All refreshed/frozen comparisons use the SAME candidate planner. It is not the original300-candidate grouping search.','Compare every1/2/4/8/16/32blocks, with either current-mixture oracle information or a one-block-lagged observed mixture. The latter avoids advance knowledge. Averages weight blocks equally. “Keep most” is defined BEFORE reading results as at least90%of fresh current-mixture oracle average remote-hit savings; also report fraction of adaptation gain recovered relative to never rebuilding.','These units are mixture blocks, not seconds. Actual request arrival timing, collection delay, rebuild/transfer disruption and bytes are absent. Do not prescribe a wall-clock rebuild interval from this test; profiling-only fit times do not include migration. Periodic phase can alias alternating trajectories, so apparent nonmonotonicity must not be read as a universal recommendation.']
for name in trajectories:
 for R in [2,4]:
  for mode in ['current_mix_oracle','previous_block_mix']:
   z=[x for x in rows if x['trajectory']==name and x['R']==R and x['information']==mode];qualified=[x['period_blocks'] for x in z if x['fraction_fresh_saving']>=.9];notes.append(f'- {name},R{R},{mode}: tested intervals retaining≥90%fresh saving: {qualified}; longest tested qualifying interval {max(qualified) if qualified else "none"}. Not a general time recommendation.')
(OUT/'rebuild/NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
