"""Loopback view of saved trajectories and predictions. No fitting or physics execution."""
import json,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import urlparse,parse_qs
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src import direct_push as d
P=ROOT/'evidence-cached-adaptation-v1'
SUMMARY=json.loads((P/'summary.json').read_text());DECISIONS=json.loads((P/'decisions.json').read_text());SEALED=json.loads((P/'selections-before-outcomes.json').read_text());ROWS=d.roster()
def case_data(cell,seed,target):
 row=ROWS[cell];traces={}
 for kind,count in [('probes',2),('decisions',7)]:
  for i in range(count):
   with np.load(ROOT/f'evidence-direct-push-v1/{cell:02d}-{kind}-{i}.npz') as z:traces[f'{kind}-{i}']=z['observations'].tolist()
 return dict(row=row,seed=seed,target=target,split='fit' if row['surface']<2 else 'evaluation',traces=traces,predictions=SEALED['seeds'][str(seed)][str(cell)],outcomes=[x for x in DECISIONS if x['cell']==cell and x['seed']==seed and x['distance']==target],goal_centroid=d.target(row,target)[0].tolist(),kind='Recorded observations and saved model predictions; no live inference or new physics')
class Handler(BaseHTTPRequestHandler):
 def send(self,data,mime):
  self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def do_GET(self):
  parsed=urlparse(self.path)
  if parsed.path in ('/','/experiment.js'):
   if parsed.query:self.send_error(400);return
   name='experiment.html' if parsed.path=='/' else 'experiment.js';self.send((ROOT/'demo'/name).read_bytes(),'text/html; charset=utf-8' if name.endswith('html') else 'text/javascript');return
  if parsed.path=='/api/index' and not parsed.query:
   self.send(json.dumps(dict(rows=ROWS,summary=SUMMARY,actions=d.ACTIONS,probes=d.PROBES)).encode(),'application/json');return
  if parsed.path!='/api/case':self.send_error(404);return
  try:
   q=parse_qs(parsed.query,strict_parsing=True)
   if set(q)!= {'cell','seed','target'} or any(len(v)!=1 for v in q.values()):raise ValueError('Exact recorded selection required')
   cell,seed,target=(int(q[k][0]) for k in ('cell','seed','target'))
   if cell not in range(32) or seed not in (9101,9102,9103) or target not in (8,20,32):raise ValueError('Outside saved roster')
   self.send(json.dumps(case_data(cell,seed,target),allow_nan=False).encode(),'application/json')
  except (ValueError,KeyError):self.send_error(400,'Invalid recorded selection')
if __name__=='__main__':
 server=HTTPServer(('127.0.0.1',8769),Handler);print('Recorded physics experiment: http://127.0.0.1:8769',flush=True);server.serve_forever()
