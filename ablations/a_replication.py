from common import *
def main():
 p,q,langs,m=data();w=np.ones(len(langs))/len(langs);rows=[];folder=OUT/'A';plt=plot();fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,g in zip(axes,[4,8]):
  groups=baseline_groups(g,langs);aware=layout(p,w,groups);default=default_plan(p,g)
  for policy in ['blind','aware_blind_copies','aware']:
   line=[]
   for R in [0,1,2,4,8,16]:
    loc=default if policy=='blind' else aware;ex=copies(p,w,groups,loc,R,blind=policy!='aware');r={'gpus':g,'R':R,'policy':policy,'memory_overhead':R/(p.shape[2]//g),**record(evaluate(q,w,groups,loc,ex,blind=policy=='blind'))};rows.append(r);line.append(r)
   ax.plot([x['memory_overhead'] for x in line],[x['traffic_cut_pct'] for x in line],marker='o',label=policy)
  ax.set(title=f'{g} simulated GPUs',xlabel='Extra expert memory / base memory',ylabel='Remote expert-hit reduction (%)');ax.grid(alpha=.3)
 axes[0].legend(fontsize=8);fig.tight_layout();folder.mkdir(parents=True,exist_ok=True);fig.savefig(folder/'frontier.png',dpi=160);save_table(folder,rows)
 notes=['# A: replication budget','Executed CPU simulation on supplied Qwen3/Wikipedia counts. Fit first half, evaluate second. Same replica count per GPU/layer. Traffic reduction is relative to uniform language-blind default (local share 1/G), not the original home-conditioned default. Expert-hit traffic proxy is not measured bytes or latency.']
 for g in [4,8]:
  for R in [0,2,4,16]:
   d={x['policy']:x for x in rows if x['gpus']==g and x['R']==R};notes.append(f"- G={g}, R={R}: aware vs blind traffic-cut advantage {d['aware']['traffic_cut_pct']-d['blind']['traffic_cut_pct']:.3f} percentage points; language-specific copies add {d['aware']['traffic_cut_pct']-d['aware_blind_copies']['traffic_cut_pct']:.3f} points. Aware mean imbalance {d['aware']['imbalance_mean']:.3f}.")
 (folder/'NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
if __name__=='__main__':main()
