"""Presentation-only rendering from the saved balance table; no simulations."""
from common import *
def render_balance():
 rows=list(csv.DictReader((OUT/'balance8/table.csv').open()));plt=plot();fig,axes=plt.subplots(1,2,figsize=(12,4.8));labels=[];handles=[]
 for G,ax in zip([4,8],axes):
  z=[x for x in rows if x['gpus']==str(G) and x['R']=='2']
  for i,x in enumerate(z):
   name=('Greedy, cap '+x['cap']) if x['method']=='greedy' and x['cap']!='None' else ('Greedy, no cap' if x['method']=='greedy' else ('Blind copies' if x['method']=='blind' else 'Assignment, no cap'))
   h=ax.scatter(float(x['imbalance_mean']),float(x['remote_hit_reduction_pct']),s=65,marker=['o','s','^','D','P','x','*'][i],label=name)
   if G==4:handles.append(h);labels.append(name)
  ax.set(title=f'{G} simulated GPUs, R2',xlabel='Mean held-out layer imbalance',ylabel='Remote expert-hit reduction (%)');ax.grid(alpha=.3);ax.margins(x=.10,y=.10)
 fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8);fig.tight_layout(rect=(0,.13,1,1));fig.savefig(OUT/'balance8/comparison.png',dpi=160)
if __name__=='__main__':render_balance()
