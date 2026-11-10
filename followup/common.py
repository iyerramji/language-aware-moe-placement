from pathlib import Path
import sys,csv,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ablations'))
import importlib.util
_spec=importlib.util.spec_from_file_location('prior_helpers',ROOT/'ablations/common.py')
_prior=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_prior)
globals().update({k:v for k,v in vars(_prior).items() if not k.startswith('_')})
OUT=ROOT/'results/raj-followup/2026-11-09'
def write_csv(name,rows):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def metric(m):
 d=record(m);d['remote_hit_reduction_pct']=d.pop('traffic_cut_pct');return d

def get_plan(name,g=4):
 d=json.loads((ROOT/'results/organized-experiments/raj-requested/D/plans.json').read_text())[name]
 p,q,langs,m=data();groups=[[langs.index(l) for l in z] for z in d['groups']];w=np.array([d['weights'][l] for l in langs]);return groups,w,np.array(d['layout'])
