from common import *
import subprocess
metadata=json.loads((OUT/'shuffle/metadata.json').read_text());trials=read('shuffle/trials.csv');summary=read('shuffle/summary.csv');originals=read('shuffle/originals.csv')
assert len(trials)==4000;assert len(summary)==40;assert len(originals)==48
for model in ['sarvam','qwen3']:
 d=np.load(DATA/f'{model}_per_sequence.npz');assert len(set(d['seq_id']))==len(d['seq_id']);k=6 if model=='sarvam' else 8
 assert np.all(d['counts'].sum(-1,dtype=np.uint64)==k*d['n_tokens'][:,None])
 with (DATA/f'{model}_index.csv').open() as f:index=list(csv.DictReader(f))
 assert len(index)==len(d['seq_id'])
 for i,row in enumerate(index):assert row['seq_id']==d['seq_id'][i] and int(row['n_tokens'])==d['n_tokens'][i] and row['split']==d['split'][i]
for name,info in metadata.items():
 model,source,mix,weighting=name.split('_',3);records,langs,k=load(model,source,mix=='indic');groups=[[langs.index(l) for l in z] for z in info['groups']];G=4;fit,ev=records['fit'],records['eval'];fa=assignments(fit['language'],langs,groups);ea=assignments(ev['language'],langs,groups)
 assert not set(fit['ids'])&set(ev['ids']);assert info['fit_ids']==fit['ids'].tolist() and info['eval_ids']==ev['ids'].tolist()
 fc=fit['counts'] if weighting=='token_work' else fit['counts'].astype(float)/fit['tokens'][:,None,None];ec=ev['counts'] if weighting=='token_work' else ev['counts'].astype(float)/ev['tokens'][:,None,None]
 for control in (['global_request_shuffle','parallel_sentence_blocked'] if source=='flores' else ['global_request_shuffle']):
  saved=np.load(OUT/f'shuffle/{name}_{control}_assignments.npz');assert saved['fit'].shape==(100,len(fc));assert saved['eval'].shape==(100,len(ec))
  for split,orig in [('fit',fa),('eval',ea)]:
   for assigned in saved[split]:assert np.array_equal(np.bincount(assigned,minlength=4),np.bincount(orig,minlength=4))
   if control=='parallel_sentence_blocked':
    parallel=records[split]['parallel']
    for index in np.unique(parallel):
     pos=parallel==index
     for assigned in saved[split][:,pos]:assert np.array_equal(np.bincount(assigned,minlength=4),np.bincount(orig[pos],minlength=4))
  assert not np.array_equal(saved['fit'][0],saved['eval'][0])
  for t in [0,99]:
   fd=aggregate(fc,saved['fit'][t],4);ed=aggregate(ec,saved['eval'][t],4);loc=placement(fd)
   for R in [2,4]:
    out=evaluate(ed,loc,copies(fd,loc,R));row=next(x for x in trials if x['dataset']==f'{model}_{source}_{mix}' and x['weighting']==weighting and x['control']==control and int(x['trial'])==t and int(x['R'])==R);assert abs(out['local_share']-float(row['local_share']))<1e-12
for row in summary:
 z=[float(x['remote_hit_reduction_pct']) for x in trials if all(x[k]==row[k] for k in ['dataset','weighting','R','control'])];assert len(z)==100;assert abs(np.mean(z)-float(row['shuffle_mean_reduction_pct']))<1e-12
# Earlier published tracks remain unmodified.
subprocess.run(['git','diff','--exit-code','--','results/organized-experiments','results/raj-followup','analyze.py','ablations'],cwd=ROOT,check=True)
curves=read('curves/table.csv');checks=read('curves/feasibility.csv');assert len(curves)==162;assert len(checks)==824
impossible=read('curves/infeasible_caps.csv')
for x in checks:
 if x['feasible']!='True':assert any(y['dataset']==x['dataset'] and y['gpus']==x['gpus'] and y['cap']==x['cap'] and float(y['minimum_primary_imbalance_bound'])>float(x['cap']) for y in impossible)
for model,source in [('sarvam','wikipedia'),('qwen3','flores'),('sarvam','flores')]:
 for G in [4,8]:
  for R in [0,2,4]:
   expected=5-len([x for x in impossible if x['dataset']==f'{model}_{source}' and x['gpus']==str(G)])
   assert len([x for x in curves if x['dataset']==f'{model}_{source}' and x['gpus']==str(G) and x['R']==str(R) and x['method']=='greedy'])==expected
rb=read('rebuild_tokens/table.csv');assert len(rb)==72
for x in rb:assert int(x['input_tokens_between_rebuilds'])==int(x['period_blocks'])*65536
print('PASS: NPZ/CSV row alignment, unique IDs, all token/k totals, disjoint fit/eval IDs; all 100-draw group sizes; FLORES within-sentence cardinalities; independent fit/eval shuffles; replayed trial0/99 from saved assignments; all 4000 trial metrics/40 summaries; 162 curve rows;824 attempted layers and two analytically infeasible cap settings; token-unit conversions; earlier tracked results unchanged.')
print('LIMIT: frozen recorded routing counts; no real-GPU latency/bytes or reconfiguration cost. FLORES sequence lengths affect token-weighted results; equal-sequence sensitivity and sentence-blocked controls are separate. Rebuild token scale is a declared synthetic scenario, not measured temporal traffic.')
