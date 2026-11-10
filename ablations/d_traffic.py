from common import *
def main():
 p,q,langs,m=data();g=4;folder=OUT/'D';folder.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(0)
 mixes={'equal':np.ones(len(langs))/len(langs),'hindi_heavy':core.weights(langs,'hi=0.5,en=0.3'),'english_heavy':core.weights(langs,'en=0.6'),'indic_only':core.weights(langs,'en=0'),'random_1':rng.dirichlet(np.full(len(langs),.5)),'random_2':rng.dirichlet(np.full(len(langs),.5))};rows=[];cache={}
 for name,w in mixes.items():
  groups=choose_groups(p,w,g);cache[name]=(groups,layout(p,w,groups));print('fitted',name,flush=True)
 (folder/'plans.json').write_text(json.dumps({name:{'weights':dict(zip(langs,w.tolist())),'groups':[[langs[i] for i in z] for z in cache[name][0]],'layout':cache[name][1].tolist()} for name,w in mixes.items()},indent=2)+'\n')
 plt=plot()
 for R in [2,4]:
  matrix=np.zeros((6,6));adv=np.zeros((6,6))
  for i,(fit,fw) in enumerate(mixes.items()):
   groups,where=cache[fit];ex=copies(p,fw,groups,where,R)
   for j,(serve,sw) in enumerate(mixes.items()):
    met=evaluate(q,sw,groups,where,ex);dflt=default_plan(p,g);blind=evaluate(q,sw,groups,dflt,copies(p,sw,groups,dflt,R,blind=True),blind=True)
    row={'R':R,'fit_mix':fit,'serve_mix':serve,'blind_traffic_cut_pct':blind['traffic_cut_pct'],'advantage_pp':met['traffic_cut_pct']-blind['traffic_cut_pct'],**record(met)};rows.append(row);matrix[i,j]=met['traffic_cut_pct'];adv[i,j]=row['advantage_pp']
  fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(matrix,cmap='viridis');names=list(mixes);ax.set_xticks(range(6),names,rotation=35,ha='right');ax.set_yticks(range(6),names);ax.set(xlabel='Served traffic mix',ylabel='Fitted traffic mix',title=f'D: remote expert-hit reduction (%), R={R}')
  for i in range(6):
   for j in range(6):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]<matrix.mean() else 'black',fontsize=8)
  fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(folder/f'fit_by_serve_R{R}.png',dpi=160)
 save_table(folder,rows);notes=['# D: concentrated and shifted traffic','Fit group search, layouts and copies exclusively on the fit half and fit mixture. Evaluate each frozen plan on the held-out half under every served mixture. Blind reference is re-fitted for each served mixture (a strong oracle-mixture comparator), using only fit-half counts. Groups have balanced LANGUAGE COUNTS as in analyze.py, not balanced request or expert load. Zero-weight English remains in group search but contributes no traffic.']
 for R in [2,4]:
  diag={x['serve_mix']:x for x in rows if x['R']==R and x['fit_mix']==x['serve_mix']}
  for name in ['equal','hindi_heavy','english_heavy','indic_only']:
   x=diag[name];notes.append(f"- R={R}, {name}: aware traffic cut {x['traffic_cut_pct']:.3f}%, advantage over blind {x['advantage_pp']:.3f} points; mean imbalance {x['imbalance_mean']:.3f}.")
  worst=max((diag[x['serve_mix']]['traffic_cut_pct']-x['traffic_cut_pct'],x['fit_mix'],x['serve_mix']) for x in rows if x['R']==R);notes.append(f'- R={R}: worst mismatch loss versus matched-fit plan {worst[0]:.3f} percentage points ({worst[1]} → {worst[2]}). These are static mixture changes, not temporal adaptation or migration-cost measurements.')
 (folder/'NOTES.md').write_text('\n\n'.join(notes)+'\n');print('\n'.join(notes))
if __name__=='__main__':main()
