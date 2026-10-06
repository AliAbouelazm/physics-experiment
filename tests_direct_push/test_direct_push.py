import unittest
import numpy as np
from shapely.geometry import Point
from src import direct_push as d
class DirectPushTests(unittest.TestCase):
 def test_full_roster(self):
  rows=d.roster();self.assertEqual(len(rows),32);self.assertEqual(len(set(r['seed'] for r in rows)),32)
  for i in range(8):
   for j in (8,16,24):self.assertEqual(rows[i]['desired_observation'],rows[i+j]['desired_observation'])
 def test_standoff(self):
  for r in d.roster():
   o=np.array(r['desired_observation']);self.assertAlmostEqual(d.geo.shape(o[2:]).distance(Point(o[:2])),17)
 def test_commands_bounded(self):
  for r in d.roster():
   for a in d.ACTIONS+d.PROBES:
    c=d.commands(r,a);self.assertEqual(c.shape,(240,2));self.assertLessEqual(np.linalg.norm(np.diff(np.vstack([r['desired_observation'][:2],c]),axis=0),axis=1).max(),2+1e-10);self.assertTrue(((c>=24)&(c<=488)).all())
 def test_hold(self):
  r=d.roster()[0];self.assertTrue(np.all(d.commands(r,d.ACTIONS[0])==r['desired_observation'][:2]))
 def test_target_independent(self):
  r=d.roster()[0];o=np.array(r['desired_observation']);self.assertAlmostEqual(d.loss(o,d.target(r,20)),1);self.assertEqual(d.target(r,20)[1],o[4])
 def test_wrapped_angle(self):
  r=d.roster()[0];o=np.array(r['desired_observation']);g=d.target(r,8);o[4]+=2*np.pi;self.assertAlmostEqual(d.loss(o,g),.16)
 def test_identical_probe_uninformative(self):
  o=np.array(d.roster()[0]['desired_observation']);x={'observations':np.repeat(o[None],241,axis=0),'contacts':np.ones(240,dtype=bool)};self.assertFalse(d.informative([x],[x]))
 def test_no_contact_uninformative(self):
  o=np.array(d.roster()[0]['desired_observation']);x={'observations':np.repeat(o[None],241,axis=0),'contacts':np.zeros(240,dtype=bool)};y={'observations':x['observations']+2,'contacts':np.ones(240,dtype=bool)};self.assertFalse(d.informative([x],[y]))

 def test_geometry_comparator_removes_known_target_headroom(self):
  contexts=[]
  for r in d.roster():
   for distance in d.DISTANCES:
    costs=[1.]*7;costs[d.DISTANCES.index(distance)]=0.
    contexts.append(dict(orientation=r['orientation'],surface=r['surface'],physics_id=r['physics_id'],distance=distance,costs=costs))
  groups,mean=d.geometry_comparator(contexts)
  self.assertEqual(len(groups),24);self.assertEqual(mean,0)
  self.assertGreater(np.mean([c['costs'] for c in contexts],axis=0).min(),.6)
  with self.assertRaises(ValueError):d.geometry_comparator(contexts[:-1])
 def test_geometry_comparator_averages_physics_not_individual_minima(self):
  contexts=[]
  for r in d.roster():
   for distance in d.DISTANCES:
    costs=[2.]*7;costs[r['physics_id']%2]=0.
    contexts.append(dict(orientation=r['orientation'],surface=r['surface'],physics_id=r['physics_id'],distance=distance,costs=costs))
  groups,mean=d.geometry_comparator(contexts)
  self.assertEqual(mean,1);self.assertTrue(all(g['action']==0 for g in groups))
