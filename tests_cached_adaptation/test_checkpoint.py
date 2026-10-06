import inspect,unittest
import numpy as np
import torch
from src import cached_adaptation as m
from src import recorded_data as r
class CheckpointTests(unittest.TestCase):
 def test_probe_only_api(self):
   self.assertEqual(list(inspect.signature(m.predict).parameters),['model','probe_y'])
   self.assertEqual(list(inspect.signature(m.choices).parameters),['pred'])
   with self.assertRaises(ValueError):m.ridge(np.zeros((7,3)),np.zeros((7,3)))
 def test_three_inputs_115_parameters(self):
   model=m.Model(9101);self.assertEqual(sum(p.numel() for p in model.parameters()),115)
   self.assertEqual(m.features(m.ACTIONS).shape,(7,3))
   np.testing.assert_array_equal(m.features([(.25,20)]),[[1/6,.5,1/12]])
 def test_fixed_seed_initialization(self):
   a=m.Model(9101);b=m.Model(9101);c=m.Model(9102)
   self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
   self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a.parameters(),c.parameters())))
   self.assertEqual(m.SEEDS,(9101,9102,9103))
 def test_ridge_three_scalar_regressions(self):
   y=np.array([[1.,2.,3.],[4.,5.,6.]]);base=np.zeros((2,3));expected=(.25*y[0]+y[1])/1.1625
   np.testing.assert_allclose(m.ridge(y,base),expected,rtol=0,atol=1e-14)
 def test_per_cell_reset_and_frozen_weights(self):
   model=m.Model(9101);before={k:v.clone() for k,v in model.state_dict().items()};y=np.array([[1.,2.,3.],[2.,4.,6.]])
   a,b,_=m.predict(model,y);m.predict(model,-y);again,b2,_=m.predict(model,y)
   np.testing.assert_array_equal(a['adapted'],again['adapted']);np.testing.assert_array_equal(b,b2)
   self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
   for v in a.values():np.testing.assert_array_equal(v[0],np.zeros(3))
 def test_no_information_simple_same_across_seed(self):
   y=np.ones((2,3));a,_,_=m.predict(m.Model(9101),y);b,_,_=m.predict(m.Model(9103),y)
   np.testing.assert_array_equal(a['simple'],b['simple'])
 def test_loss_matches_original_metric(self):
   row=r.d.roster()[0];obs=np.array(row['desired_observation']);obs[2:4]+=[4,2];obs[4]+=.1;y=r.response(row,obs)
   self.assertAlmostEqual(m.losses(y[None],20)[0],r.d.loss(obs,r.d.target(row,20)),places=12)
 def test_choices_all_targets_ties(self):
   self.assertEqual(m.choices(np.zeros((7,3))),{'8':0,'20':0,'32':0})
 def test_gate_needs_prediction_and_all_comparators(self):
   f={'prediction_mse':1.,'mean_loss':1.};a={'prediction_mse':.8,'mean_loss':.8};s={'prediction_mse':.9,'mean_loss':.9}
   g=m.gates(f,a,s,.85,[(1.,.8,.9)]*4);self.assertTrue(all(g[k] for k in ('prediction','decision','orbits','geometry_aware')))
   a['prediction_mse']=.99;self.assertFalse(m.gates(f,a,s,.85,[(1.,.8,.9)]*4)['prediction'])
