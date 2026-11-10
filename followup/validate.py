from common import *
import itertools
# Separate outputs and immutable prior data.
root=ROOT/'results/organized-experiments';preserved=ROOT/'results/preserved-mentor-ablations/2026-11-09'
assert (root/'raj-requested/submission.zip').read_bytes()==(preserved/'submission.zip').read_bytes()
source=np.load(ROOT/'results/routing_qwen3_30b_a3b.npz');p,q,langs,m=data();rng=np.random.default_rng(3)
# Independent aggregate-profile expectation matches explicit origin/expert loops for arbitrary fixed placements.
for name in ['indic_only','equal']:
 groups,w,where=get_plan(name);G=4;pi=np.array([w[g].sum() for g in groups]);aggregate=(w[:,None,None]*q).sum(0);qm=np.repeat(aggregate[None],len(w),axis=0)
 assert abs(pi.sum()-1)<1e-12
 np.testing.assert_allclose((w[:,None,None]*qm).sum(0),aggregate)
 for R in [2,4]:
  ex=copies(p,w,groups,where,R);met=evaluate(qm,w,groups,where,ex);local=[];loads=[]
  for l in range(48):
   load=np.zeros(G);hits=0.
   for g in range(G):
    for e in range(128):
     demand=pi[g]*aggregate[l,e];destination=g if where[l,e]==g or ex[l,g,e] else where[l,e];load[destination]+=demand
     if destination==g:hits+=demand
   local.append(hits);loads.append(load)
  np.testing.assert_allclose(local,met['per_layer_local']);np.testing.assert_allclose(loads,met['per_layer_load']);assert np.allclose(np.sum(loads,axis=1),1)
# Exhaust all group-size-preserving request allocations on a tiny invented TEST fixture (not research data).
request=np.array([[.8,.2],[.1,.9],[.5,.5],[.2,.8]]);group_mass=.5;expected=0
for assigned in itertools.combinations(range(4),2):
 other=[i for i in range(4) if i not in assigned];expected+=(request[list(assigned),0].sum()+request[other,1].sum())/4
expected/=6;analytic=.5*request.mean(0)[0]+.5*request.mean(0)[1];assert abs(expected-analytic)<1e-12
# All requested grid points and replay checks.
read=lambda f:list(csv.DictReader((OUT/f).open()))
sh=read('shuffle/table.csv');assert len(sh)==16;assert len(read('shuffle/whole-language-shuffles.csv'))==400
bb=read('balance8/table.csv');assert len(bb)==42;checks=read('balance8/layer-checks.csv');assert len(checks)==480;assert all(x['feasible']=='True' for x in checks)
placements=np.load(OUT/'balance8/placements.npz')
for key in placements.files:
 G=int(key.split('_')[0]);cap=None if key.endswith('None') else float(key.split('_')[1]);groups=baseline_groups(G,langs);w=np.ones(10)/10;where=placements[key]
 for l in range(48):
  load=(w[:,None]*p[:,l]).sum(0);gpu_load=np.array([load[where[l]==g].sum() for g in range(G)]);assert all((where[l]==g).sum()==128//G for g in range(G));assert cap is None or gpu_load.max()/gpu_load.mean()<=cap+1e-6
# Match earlier greedy implementation's exact metric values on four GPUs.
old=list(csv.DictReader((ROOT/'results/organized-experiments/raj-requested/B/table.csv').open()))
for row in bb:
 if row['gpus']=='4' and row['method']=='greedy':
  ref=next(x for x in old if x['method']=='greedy' and x['cap']==row['cap'] and x['R']==row['R']);assert abs(float(row['local_share'])-float(ref['local_share']))<1e-12
rb=read('rebuild/table.csv');blocks=read('rebuild/block-results.csv');assert len(rb)==72;assert len(blocks)==2304
for row in rb:
 sub=[x for x in blocks if all(x[k]==row[k] for k in ['trajectory','R','period_blocks','information'])];assert len(sub)==32;assert abs(np.mean([float(x['remote_hit_reduction_pct']) for x in sub])-float(row['reduction_pct']))<1e-12
 assert abs(float(row['fraction_fresh_saving'])-float(row['reduction_pct'])/float(row['fresh_reduction_pct']))<1e-12
for row in sh+bb:
 assert 0<=float(row['local_share'])<=1;assert np.isfinite(float(row['remote_hit_reduction_pct']))
print('PASS: conserved shuffled-expectation demand and fixed group masses; exhaustive toy request-permutation expectation; all 16 shuffle rows / 400 packet permutations; 480 feasible greedy layers; 42 balance rows; exact four-GPU prior-greedy parity; 72 rebuild settings / 2304 block records; prior A-D archive unchanged.')
print('LIMIT: no per-request routing traces; shuffle realization/uncertainty unavailable. Rebuild trajectories synthetic; no wall-clock deployment interval or actual bytes/latency inferred.')
