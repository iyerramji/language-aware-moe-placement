from common import *
s=read('shuffle/summary.csv');plt=plot();fig,axes=plt.subplots(1,2,figsize=(12,4.8));names=['sarvam_wikipedia','qwen3_flores','sarvam_flores'];x=np.arange(3);width=.34
for mix,ax in zip(['full','indic'],axes):
 z=[next(r for r in s if r['dataset']==f'{n}_{mix}' and r['weighting']=='token_work' and r['R']=='2' and r['control']=='global_request_shuffle') for n in names];orig=np.array([float(r['original_reduction_pct']) for r in z]);null=np.array([float(r['shuffle_mean_reduction_pct']) for r in z]);lo=np.array([float(r['shuffle_p025']) for r in z]);hi=np.array([float(r['shuffle_p975']) for r in z]);blind=[float(r['blind_reduction_pct']) for r in z]
 ax.bar(x-width/2,orig,width,label='Original groups',color='#147d86');ax.bar(x+width/2,null,width,label='Shuffled groups: mean',color='#d28b32',yerr=np.stack([null-lo,hi-null]),capsize=3);ax.scatter(x,blind,color='#222',marker='x',label='Blind uniform origins');ax.set_xticks(x,['Sarvam\nWikipedia','Qwen3\nFLORES','Sarvam\nFLORES']);ax.set(title='Full mix' if mix=='full' else 'Indic-only',ylabel='Remote expert-hit reduction (%)');ax.grid(axis='y',alpha=.3)
axes[0].legend(fontsize=8);fig.suptitle('Exact recorded-sequence shuffle: 4 GPUs, R2, 100 draws; token-work weighting');fig.tight_layout();fig.savefig(OUT/'shuffle/exact_shuffle.png',dpi=160)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for mix,ax in zip(['full','indic'],axes):
 for i,weight in enumerate(['token_work','equal_sequence']):
  z=[next(r for r in s if r['dataset']==f'{n}_{mix}' and r['weighting']==weight and r['R']=='2' and r['control']=='global_request_shuffle') for n in names];ax.bar(x+(i-.5)*.34,[float(r['original_minus_shuffle_mean_pp']) for r in z],.34,label=weight.replace('_',' '))
 ax.set_xticks(x,['Sarvam Wiki','Qwen3 FLORES','Sarvam FLORES']);ax.set(title=mix,ylabel='Original minus shuffled mean (points)');ax.axhline(0,color='grey');ax.grid(axis='y',alpha=.3)
axes[0].legend();fig.tight_layout();fig.savefig(OUT/'shuffle/length_sensitivity.png',dpi=160)
print('Rendered shuffle and length-sensitivity figures')
