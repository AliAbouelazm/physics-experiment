"""Real browser interaction and screenshots of the recorded experiment."""
import argparse,hashlib,json,socket,subprocess,sys,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--server-python',default=sys.executable);parser.add_argument('--chromium',default='/usr/bin/chromium');args=parser.parse_args()
 OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
 with socket.socket() as sock:sock.bind(('127.0.0.1',8769))
 with (OUT/'server.log').open('w') as log:
  server=subprocess.Popen([args.server_python,'demo/experiment_server.py'],cwd=ROOT,stdout=log,stderr=log)
  try:
   for _ in range(40):
    if server.poll() is not None:raise RuntimeError('Server exited')
    try:urllib.request.urlopen('http://127.0.0.1:8769/api/index',timeout=1).close();break
    except OSError:time.sleep(.2)
   else:raise RuntimeError('Server not ready')
   with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.chromium,headless=True);page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1);errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto('http://127.0.0.1:8769');page.wait_for_selector('html[data-ready="true"]')
    assert page.locator('#cell option').count()==32 and page.locator('canvas').count()==3
    assert page.evaluate('experimentState.cell')==2 and page.evaluate('experimentState.step')==240
    case=page.request.get('http://127.0.0.1:8769/api/case?cell=2&seed=9101&target=20').json()
    import numpy as np
    with np.load(ROOT/'evidence-direct-push-v1/02-decisions-0.npz') as z:assert case['traces']['decisions-0']==z['observations'].tolist()
    for method in ('frozen','adapted','simple'):
     out=next(x for x in case['outcomes'] if x['method']==method);assert f"Loss {out['loss']:.3f}" in page.locator('#metrics-'+method).inner_text()
    page.screenshot(path=str(OUT/'physics-experiment-desktop.png'),full_page=True)
    page.click('#play');page.wait_for_function('experimentState.step>2');page.click('#play');assert page.evaluate('experimentState.step')<240
    page.locator('#step').fill('80');page.locator('#step').dispatch_event('input');assert page.evaluate('experimentState.step')==80
    page.select_option('#episode','probes-1');assert page.evaluate('experimentState.episode')=='probes-1'
    for cell,seed,target in [(0,9102,8),(15,9103,32),(31,9101,20)]:
     page.select_option('#cell',str(cell));page.select_option('#seed',str(seed));page.select_option('#target',str(target));page.wait_for_function(f"document.documentElement.dataset.ready==='true' && experimentState.cell==={cell} && experimentState.seed==={seed} && experimentState.target==={target}")
    page.select_option('#episode','decision');page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(OUT/'physics-experiment-mobile.png'),full_page=True)
    rejected={}
    for path in ('/.git/config','/%2e%2e/.git/config','/evidence-cached-adaptation-v1/model-9101.pt','/api/case?cell=32&seed=9101&target=20','/api/case?cell=2&seed=9101&target=20&extra=1'):
     rejected[path]=page.request.get('http://127.0.0.1:8769'+path).status;assert rejected[path] in (400,404)
    assert not errors;browser.close()
   (OUT/'browser.json').write_text(json.dumps(dict(status='passed',browser='Chromium via Playwright',recorded_default=dict(cell=2,seed=9101,target=20),recorded_trace_exact=True,cases_checked=[0,2,15,31],controls=['case','seed','target','episode','play/pause','timeline'],mobile_width=390,page_errors=errors,rejected_routes=rejected,screenshots={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.png')}),indent=2)+'\n')
  finally:
   server.terminate()
   try:server.wait(timeout=5)
   except subprocess.TimeoutExpired:server.kill();server.wait()
   with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',8769))!=0
if __name__=='__main__':main()
