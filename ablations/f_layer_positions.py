"""F: replica-location sensitivity for an English/Indic layout. Existing expert-copy placement only."""
from common import *

def main():
 p,q,langs,m=data();w=np.ones(len(langs))/len(langs);L,E=p.shape[1:];plans=json.loads((OUT/'E/plans.json').read_text());folder=OUT/'F';folder.mkdir(parents=True,exist_ok=True);rows=[];allocation={};plt=plot();fig,axes=plt.subplots(1,2,figsize=(12,4))
 for g,ax in zip([4,8],axes):
  groups=[[langs.index(l) for l in z] for z in plans[f'{g}_equal_english_indic']['groups']];where=layout(p,w,groups)
  bands={'uniform':np.full((L,g),2),'early_8':np.zeros((L,g),int),'middle_32':np.zeros((L,g),int),'late_8':np.zeros((L,g),int),'boundaries_16':np.zeros((L,g),int)}
  bands['early_8'][:8]=12;bands['middle_32'][8:40]=3;bands['late_8'][40:]=12;bands['boundaries_16'][:8]=6;bands['boundaries_16'][40:]=6
  # Match 96 copies/GPU for every 8-layer window. Choose best window on FIT data exclusively.
  windows=[]
  for start in range(L-7):
   b=np.zeros((L,g),int);b[start:start+8]=12;ex=copies(p,w,groups,where,b);fit=evaluate(p,w,groups,where,ex)['local_share'];windows.append((fit,start,b))
  _,start,b=max(windows,key=lambda x:(x[0],-x[1]));bands['fit_selected_window8']=b
  for name,b in bands.items():
   assert np.all(b.sum(0)==96);ex=copies(p,w,groups,where,b);assert np.all(ex.sum((0,2))==96);met=evaluate(q,w,groups,where,ex);fit=evaluate(p,w,groups,where,ex);rows.append({'gpus':g,'variant':name,'total_copies_per_gpu':96,'fit_local_share':fit['local_share'],'selected_window_start':start+1 if name=='fit_selected_window8' else '',**record(met)});allocation[f'{g}_{name}']=b.tolist()
  z=[x for x in rows if x['gpus']==g];ax.bar(range(len(z)),[x['traffic_cut_pct'] for x in z]);ax.set_xticks(range(len(z)),[x['variant'] for x in z],rotation=40,ha='right');ax.set(title=f'{g} simulated GPUs',ylabel='Remote expert-hit reduction (%)');ax.grid(axis='y',alpha=.3)
 fig.suptitle('F: where replicas go in an English/Indic plan');fig.tight_layout();fig.savefig(folder/'layer_positions.png',dpi=160);save_table(folder,rows);(folder/'allocations.json').write_text(json.dumps(allocation,indent=2)+'\n')
 notes=['# F: English/Indic replica position sensitivity','This tests existing expert-copy allocation. Router decisions and model weights are unchanged.','English/Indic groups come from E, fitted on training counts and held fixed across layers. All variants have96replicas per GPU across48layers and unchanged128logical experts per layer. Early/late windows have12copies per layer; middle32has3; both boundaries have6in each of16layers; uniform has2in all48. The8-layer sliding window is selected using fit data only. No new home migration between layers is assumed.']
 for g in [4,8]:
  z=[x for x in rows if x['gpus']==g];uniform=next(x for x in z if x['variant']=='uniform');mid=next(x for x in z if x['variant']=='middle_32');win=next(x for x in z if x['variant']=='fit_selected_window8');notes.append(f"- G={g}: uniform {uniform['traffic_cut_pct']:.3f}%, middle-only {mid['traffic_cut_pct']:.3f}% (delta {mid['traffic_cut_pct']-uniform['traffic_cut_pct']:.3f} points); fit-selected window starts at layer {win['selected_window_start']}, gives {win['traffic_cut_pct']:.3f}% reduction. These compare equal total copies, not equal copies per layer.")
 (folder/'NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
if __name__=='__main__':main()
