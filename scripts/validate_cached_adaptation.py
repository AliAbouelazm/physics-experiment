"""Read-only checkpoint/policy reconstruction and isolation audit; never fit or simulate."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from src import cached_adaptation as m
from src import recorded_data as r
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence-cached-adaptation-v1'
def run():
 s=json.loads((OUT/'summary.json').read_text());sealed=json.loads((OUT/'selections-before-outcomes.json').read_text());history=json.loads((OUT/'fit-history.json').read_text());access=json.loads((OUT/'cache-access.json').read_text());decisions=json.loads((OUT/'decisions.json').read_text())
 assert s['completed_fits']==3 and s['updates_per_seed']==300 and s['parameters']==115 and s['native_steps']==s['native_resets']==0
 assert not s['positive_signal'] and s['qualification_remains_failed']
 assert hashlib.sha256((OUT/'selections-before-outcomes.json').read_bytes()).hexdigest()==s['selection_file_sha256']
 assert all(x['after_seal'] for x in access if x['cell'] in m.EVAL_CELLS and x['kind']=='decisions')
 assert len([x for x in access if x['cell'] in m.EVAL_CELLS and x['kind']=='decisions'])==112
 max_error=0.;count=0
 for seed in m.SEEDS:
  assert len(history[str(seed)])==300 and [x['step'] for x in history[str(seed)]]==list(range(1,301))
  model=m.Model(seed);path=OUT/f'model-{seed}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==sealed['model_sha256'][str(seed)];model.load_state_dict(torch.load(path,weights_only=True));model.eval()
  for row in r.d.roster():
   probes=[]
   for i in range(2):
    with np.load(ROOT/f'evidence-direct-push-v1/{row["cell"]:02d}-probes-{i}.npz') as z:probes.append(r.response(row,z['observations'][-1]))
   predictions,_,_=m.predict(model,np.array(probes));entry=sealed['seeds'][str(seed)][str(row['cell'])]
   for method,pred in predictions.items():
    max_error=max(max_error,float(np.max(np.abs(pred-np.array(entry['predictions'][method])))));assert m.choices(pred)==entry['choices'][method];count+=1
 assert max_error<1e-12
 assert len(decisions)==2304 and len({(x['seed'],x['cell'],x['distance'],x['method']) for x in decisions})==2304
 assert all(x['charged_controls']==726 for x in decisions)
 result=dict(checkpoint_policy_reconstructions=count,max_prediction_difference=max_error,heldout_candidate_reads_before_seal=0,heldout_candidate_reads_after_seal=112,decision_records=len(decisions),charged_reset_range=[min(x['charged_resets'] for x in decisions),max(x['charged_resets'] for x in decisions)],fit_updates_verified=900,validation_fits=0,native_steps=0)
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':run()
