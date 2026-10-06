"""Public canonicalization for read-only reconstruction of saved endpoints."""
import numpy as np
from src import direct_push as d

def response(row,obs):
    initial=np.array(row['desired_observation']);_,normal=d.geo.contact_geometry(initial,row['surface']);tangent=d.geo.rotation(initial[4])@np.array([0.,1.]);delta=d.centroid(obs)-d.centroid(initial);sign=-d.geo.SURFACES[row['surface']][2]
    return np.array([delta@(-normal)/20,delta@tangent/20,50*sign*d.geo.angle_difference(obs[4],initial[4])/20])
