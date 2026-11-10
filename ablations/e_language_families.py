"""E: Family and English/Indic home-GPU grouping; existing experts/routers unchanged."""
from common import *
import itertools
FAMILY={'en':'English','hi':'Indo-Aryan','bn':'Indo-Aryan','mr':'Indo-Aryan','gu':'Indo-Aryan','pa':'Indo-Aryan','ta':'Dravidian','te':'Dravidian','kn':'Dravidian','ml':'Dravidian'}

def partition(items,sizes):
 if len(sizes)==1:
  yield [list(items)];return
 for block in itertools.combinations(items,sizes[0]):
  rest=tuple(x for x in items if x not in block)
  for tail in partition(rest,sizes[1:]):
   groups=[list(block)]+tail
   # Equal-sized blocks are interchangeable GPUs; canonicalize to avoid duplicate enumeration.
   if all(sizes[i]!=sizes[j] or groups[i][0]<groups[j][0] for i in range(len(sizes)) for j in range(i+1,len(sizes))):yield groups

def best(p,w,candidates):
 bestscore=-1;chosen=None;n=0
 for groups in candidates:
  score=0.
  for l in range(p.shape[1]):
   prof=core.profits(p,groups,w,l);where=core.place_layer(prof,p.shape[2]//len(groups));score+=prof[np.arange(p.shape[2]),where].sum()
  n+=1
  if score>bestscore+1e-12:bestscore=score;chosen=groups
 assert chosen is not None
 return chosen,bestscore/p.shape[1],n

def families(ia,dr,en,g):
 ia_sizes,dr_sizes=([3,2],[4]) if g==4 else ([2,1,1,1],[2,1,1])
 for a in partition(tuple(sorted(ia)),ia_sizes):
  for b in partition(tuple(sorted(dr)),dr_sizes):yield [[en]]+a+b

def english_indic(en,other,g):
 # English gets one home GPU; all Indic languages share remaining pool. Fit chooses subgroups inside pool.
 sizes=([3,3,3] if g==4 else [2,2,1,1,1,1,1])
 yield from ([[en]]+groups for groups in partition(tuple(sorted(other)),sizes))

def main():
 p,q,langs,m=data();en=langs.index('en');ia=[i for i,l in enumerate(langs) if FAMILY[l]=='Indo-Aryan'];dr=[i for i,l in enumerate(langs) if FAMILY[l]=='Dravidian'];indic=[i for i in range(len(langs)) if i!=en]
 folder=OUT/'E';folder.mkdir(parents=True,exist_ok=True);rows=[];plans={};null=[];rng=random.Random(71);all_labels=[x for x in itertools.combinations(indic,5) if set(x)!=set(ia)];shuffled=rng.sample(all_labels,20)
 mixes={'equal':np.ones(len(langs))/len(langs),'indic_only':core.weights(langs,'en=0'),'hindi_heavy':core.weights(langs,'hi=0.5,en=0.3'),'english_heavy':core.weights(langs,'en=0.6')}
 for g in [4,8]:
  for mix,w in mixes.items():
   fg,fs,fn=best(p,w,families(ia,dr,en,g));eg,es,ec=best(p,w,english_indic(en,indic,g));rg=baseline_groups(g,langs) if mix=='equal' else choose_groups(p,w,g)
   # Control for same group sizes and English isolation: shuffle Indic identities into fixed sizes; fit-select 300 candidates.
   sizes=[len(z) for z in fg[1:]];rr=random.Random(19);controls=[]
   for _ in range(300):
    order=indic.copy();rr.shuffle(order);pos=0;groups=[[en]]
    for size in sizes:groups.append(order[pos:pos+size]);pos+=size
    controls.append(groups)
   cg,cs,cn=best(p,w,controls)
   chosen={'family':fg,'english_indic':eg,'routing_derived':rg,'english_same_sizes_control':cg,'language_blind':rg}
   for name,groups in chosen.items():
    where=default_plan(p,g) if name=='language_blind' else layout(p,w,groups);plans[f'{g}_{mix}_{name}']={'groups':[[langs[i] for i in z] for z in groups],'fit_mix':dict(zip(langs,w.tolist())),'layout':where.tolist()}
    for R in [0,2,4]:
     ex=copies(p,w,groups,where,R,blind=name=='language_blind');met=evaluate(q,w,groups,where,ex,blind=name=='language_blind');rows.append({'gpus':g,'mix':mix,'policy':name,'R':R,'token_imbalance':float(max(w[z].sum() for z in groups)*g) if name!='language_blind' else 1.,**record(met)})
   # Shuffled-family control on equal traffic only: all have same family sizes, GPU allocation, and within-family fit optimization.
   if mix=='equal':
    for idx,label in enumerate(shuffled):
     other=[i for i in indic if i not in label];groups,score,n=best(p,w,families(list(label),other,en,g));where=layout(p,w,groups)
     for R in [0,2,4]:
      met=evaluate(q,w,groups,where,copies(p,w,groups,where,R));null.append({'gpus':g,'shuffle':idx,'R':R,'fit_objective':score,'pseudo_ia':','.join(langs[i] for i in label),**record(met)})
   save_table(folder,rows);(folder/'plans.json').write_text(json.dumps(plans,indent=2)+'\n');print('finished',g,mix,flush=True)
 with (folder/'shuffle-control.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(null[0]));writer.writeheader();writer.writerows(null)
 # Routing distance is descriptive and excludes English from both within/between-family comparisons.
 dist=np.array([[core.js_bits(q[a],q[b]).mean() for b in indic] for a in indic]);within=[];between=[]
 for i,a in enumerate(indic):
  for j,b in enumerate(indic):
   if j<=i:continue
   (within if FAMILY[langs[a]]==FAMILY[langs[b]] else between).append(float(dist[i,j]))
 diagnostics={'within_family_pairwise_js_bits':float(np.mean(within)),'between_family_pairwise_js_bits':float(np.mean(between)),'within_pairs':len(within),'between_pairs':len(between),'family_mapping':FAMILY,'family_source':'https://censusindia.gov.in/nada/index.php/catalog/42458/download/46089/C-16_25062018.pdf'};(folder/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
 plt=plot();fig,axes=plt.subplots(1,2,figsize=(12,4))
 for g,ax in zip([4,8],axes):
  for name in chosen:
   z=[x for x in rows if x['gpus']==g and x['mix']=='equal' and x['policy']==name];ax.plot([x['R'] for x in z],[x['traffic_cut_pct'] for x in z],marker='o',label=name)
  ax.set(title=f'{g} simulated GPUs, equal traffic',xlabel='Extra expert copies / GPU / layer',ylabel='Remote expert-hit reduction (%)');ax.grid(alpha=.3)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(folder/'family_frontier.png',dpi=160)
 notes=['# E: language-family and English/Indic grouping','Executed on Qwen3/Wikipedia marginal routing counts only. Indo-Aryan:Hindi,Bengali,Marathi,Gujarati,Punjabi. Dravidian:Tamil,Telugu,Kannada,Malayalam. English held separate as an experimental control, not classified as Indo-Aryan. Family classification from Census of India, Statement9.','Four GPUs:1English home,2Indo-Aryan homes,1Dravidian home; family subgroup sizes1/3/2/4. Eight GPUs:1English,4Indo-Aryan,3Dravidian homes. English/Indic-only pooling keeps English on one GPU and fit-optimizes Indic partitions among the other GPUs. Every GPU retains E/G primary expert slots; same R copies per GPU/layer. All experts remain accessible; no router is changed. This is home-assignment grouping, not an architectural language-specific expert restriction.','The different grouping sizes and home workload can confound raw comparisons. The same-sizes control fixes English isolation and group sizes but removes family restrictions. Twenty seeded shuffled-family labelings preserve all family sizes and within-family optimization, excluding the true mapping. They are an exploratory control, not exhaustive statistical proof. Grouping uses fit counts; metrics use held-out counts. No network bytes or latency measured.']
 for g in [4,8]:
  fam=next(x for x in rows if x['gpus']==g and x['mix']=='equal' and x['policy']=='family' and x['R']==2);control=next(x for x in rows if x['gpus']==g and x['mix']=='equal' and x['policy']=='english_same_sizes_control' and x['R']==2);pool=next(x for x in rows if x['gpus']==g and x['mix']=='equal' and x['policy']=='english_indic' and x['R']==2);nulls=[x['traffic_cut_pct'] for x in null if x['gpus']==g and x['R']==2]
  notes.append(f"- G={g}, R2 equal: family remote-hit reduction {fam['traffic_cut_pct']:.3f}%, imbalance {fam['imbalance_mean']:.3f}; advantage versus English-separated same-sizes control {fam['traffic_cut_pct']-control['traffic_cut_pct']:.3f} percentage points. English/Indic pooling {pool['traffic_cut_pct']:.3f}% reduction, imbalance {pool['imbalance_mean']:.3f}. Family beats {sum(v<fam['traffic_cut_pct'] for v in nulls)}/20 shuffled controls; shuffle range {min(nulls):.3f}–{max(nulls):.3f}%.")
 notes.append(f"- Held-out Indic pairwise JS: within-family {np.mean(within):.6f} bits, between-family {np.mean(between):.6f} bits. Descriptive pair means; overlapping language pairs are not independent samples.")
 (folder/'NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
if __name__=='__main__':main()
