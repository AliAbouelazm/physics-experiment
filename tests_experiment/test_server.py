import json,threading,unittest,urllib.request,urllib.error
from http.server import HTTPServer
import numpy as np
from demo import experiment_server as s
class ExperimentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=HTTPServer(('127.0.0.1',0),s.Handler);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.url=f'http://127.0.0.1:{cls.server.server_port}'
 @classmethod
 def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
 def test_complete_index(self):
  with urllib.request.urlopen(self.url+'/api/index') as r:data=json.load(r)
  self.assertEqual(len(data['rows']),32);self.assertFalse(data['summary']['positive_signal'])
 def test_recorded_trace_and_decisions(self):
  with urllib.request.urlopen(self.url+'/api/case?cell=2&seed=9101&target=20') as r:data=json.load(r)
  with np.load(s.ROOT/'evidence-direct-push-v1/02-decisions-0.npz') as z:np.testing.assert_array_equal(data['traces']['decisions-0'],z['observations'])
  self.assertEqual(len(data['outcomes']),8);self.assertEqual(data['split'],'evaluation');self.assertIn('no live inference',data['kind'])
 def test_invalid_routes_and_inputs(self):
  for path,status in [('/.git/config',404),('/api/case?cell=32&seed=9101&target=20',400),('/api/case?cell=2&seed=9999&target=20',400),('/api/case?cell=2&seed=9101&target=20&target=8',400),('/api/case?cell=2&seed=9101&target=20&physics=1',400)]:
   with self.subTest(path=path),self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(self.url+path)
   self.assertEqual(caught.exception.code,status)
 def test_page_declares_recorded_and_failed(self):
  with urllib.request.urlopen(self.url+'/') as r:html=r.read().decode()
  self.assertIn('Recorded replay',html);self.assertIn('no new physics or live inference',html);self.assertIn('Benchmark tuning has stopped',html)
 def test_fit_family_is_labeled(self):self.assertEqual(s.case_data(0,9102,8)['split'],'fit')
