from common import *
# Re-express the earlier synthetic equal-work trajectory in an explicitly specified input-token scenario.
BLOCK_TOKENS=128*512
previous=ROOT/'results/raj-followup/2026-11-09/rebuild';rows=list(csv.DictReader((previous/'table.csv').open()));blocks=list(csv.DictReader((previous/'block-results.csv').open()))
for x in rows:
 x['input_tokens_between_rebuilds']=int(x['period_blocks'])*BLOCK_TOKENS;x['input_tokens_total']=int(x['blocks'])*BLOCK_TOKENS;x['observation_delay_input_tokens']=BLOCK_TOKENS if x['information']=='previous_block_mix' else 0
for x in blocks:x['cumulative_input_tokens_end']=(int(x['block'])+1)*BLOCK_TOKENS
write('rebuild_tokens/table.csv',rows);write('rebuild_tokens/serving_windows.csv',blocks)
info={'input_tokens_per_window':BLOCK_TOKENS,'requests_per_window':128,'input_tokens_per_request':512,'total_input_tokens':32*BLOCK_TOKENS,'scenario':'Synthetic fixed-length Wikipedia-like input workload; token scale explicitly chosen, not inferred from a production trace. Existing trajectory metrics re-expressed without changing results.','source_table_sha256':hashlib.sha256((previous/'table.csv').read_bytes()).hexdigest(),'source_window_sha256':hashlib.sha256((previous/'block-results.csv').read_bytes()).hexdigest()};(OUT/'rebuild_tokens/scenario.json').write_text(json.dumps(info,indent=2)+'\n')
plt=plot();fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
from matplotlib.ticker import FuncFormatter
for name,ax in zip(['gradual','abrupt','alternating'],axes):
 for mode,title in [('current_mix_oracle','Current mix known'),('previous_block_mix','Previous window only')]:
  z=[x for x in rows if x['trajectory']==name and x['R']=='2' and x['information']==mode];ax.plot([x['input_tokens_between_rebuilds'] for x in z],[100*float(x['fraction_fresh_saving']) for x in z],marker='o',label=title)
 ax.axhline(90,color='grey',linestyle='--');ax.set(title=name,xlabel='Input tokens between rebuilds');ax.set_xscale('log',base=2);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1e6:.2g}M' if x>=1e6 else f'{x/1000:.0f}k'));ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.3)
axes[0].set_ylabel('Fresh-plan saving retained (%)');axes[0].legend(fontsize=7);fig.suptitle('Token-normalized synthetic workload; 4 GPUs, R2; no migration cost');fig.tight_layout();fig.savefig(OUT/'rebuild_tokens/update_interval_tokens.png',dpi=160)
print('Token intervals',[BLOCK_TOKENS*k for k in [1,2,4,8,16,32]])
