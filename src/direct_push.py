"""Direct-contact identification benchmark v1. No routed approach or learned model."""
import itertools, math
import numpy as np
from shapely.geometry import Point
from src import pusht_v4 as geo
from src.pusht_task import TRAIN
ACTIONS=((0.,0),)+tuple(itertools.product((.25,.75,1.5),(20,40)))
PROBES=((.25,20),(1.,20))
HORIZON=240
DISTANCES=(8.,20.,32.)

def roster():
    rows=[]
    for physics_id,physics in enumerate(TRAIN):
        for orientation,angle in enumerate((0.,math.pi/4)):
            for surface in range(4):
                obs=np.array([0.,0.,256.,236.,angle]);point,normal=geo.contact_geometry(obs,surface);obs[:2]=point+17*normal
                rows.append(dict(cell=len(rows),physics_id=physics_id,physics=list(physics),orientation=orientation,surface=surface,seed=8800+len(rows),desired_observation=obs.tolist()))
    return rows

def commands(row,action):
    """Absolute public ramp, then withdrawal and stationary target; no block feedback."""
    obs=np.array(row['desired_observation']);_,normal=geo.contact_geometry(obs,row['surface']);speed,duration=action;start=obs[:2];result=[]
    for t in range(HORIZON):
        depth=speed*min(t+1,duration)
        retreat=0 if duration==0 else 2*min(max(t+1-duration,0),40)
        result.append(start+(retreat-depth)*normal)
    return np.array(result)

def centroid(obs):return np.asarray(obs)[2:4]+geo.rotation(obs[4])@geo.CENTROID

def target(row,distance):
    obs=np.array(row['desired_observation']);_,normal=geo.contact_geometry(obs,row['surface'])
    return centroid(obs)-distance*normal,obs[4]

def loss(obs,goal):
    center,angle=goal
    return float((np.sum((centroid(obs)-center)**2)+(50*geo.angle_difference(obs[4],angle))**2)/400)

def informative(a,b):
    for x,y in zip(a,b):
        if not (x['contacts'].any() and y['contacts'].any()):continue
        xo=x['observations'];yo=y['observations']
        if np.linalg.norm(xo[:,2:4]-yo[:,2:4],axis=1).max()>=1:return True
        angle=np.arctan2(np.sin(xo[:,4]-yo[:,4]),np.cos(xo[:,4]-yo[:,4]))
        if np.abs(angle).max()>=math.pi/180:return True
    return False

def geometry_comparator(contexts):
    """Evaluator-only group optimum; never conditions on individual hidden physics."""
    groups=[]
    for orientation,surface,distance in itertools.product(range(2),range(4),DISTANCES):
        group=[c for c in contexts if (c['orientation'],c['surface'],c['distance'])==(orientation,surface,distance)]
        if len(group)!=4 or {c['physics_id'] for c in group}!=set(range(4)):
            raise ValueError('Comparator requires all four physics per geometry/target')
        means=np.mean([c['costs'] for c in group],axis=0);action=int(np.argmin(means))
        groups.append(dict(orientation=orientation,surface=surface,distance=distance,action=action,mean_loss=float(means[action]),action_means=means.tolist()))
    return groups,float(np.mean([g['mean_loss'] for g in groups]))

def score(rows,traces):
    contexts=[];pairs=[]
    for row in rows:
        decisions=traces[row['cell']]['decisions']
        for distance in DISTANCES:
            costs=[loss(d['observations'][-1],target(row,distance)) for d in decisions];best=int(np.argmin(costs))
            contexts.append(dict(cell=row['cell'],physics_id=row['physics_id'],orientation=row['orientation'],surface=row['surface'],distance=distance,costs=costs,best=best,controllable=costs[0]-costs[best]>=.1 and bool(decisions[best]['contacts'].any())))
    for orientation,surface,distance in itertools.product(range(2),range(4),DISTANCES):
        group=[c for c in contexts if (c['orientation'],c['surface'],c['distance'])==(orientation,surface,distance)]
        for a,b in itertools.combinations(group,2):
            ai,bi=a['best'],b['best'];flip=ai!=bi and a['costs'][bi]-a['costs'][ai]>=.1 and b['costs'][ai]-b['costs'][bi]>=.1
            info=informative(traces[a['cell']]['probes'],traces[b['cell']]['probes'])
            pairs.append(dict(cells=[a['cell'],b['cell']],orientation=orientation,surface=surface,distance=distance,raw_flip=flip,informative=info,counted_flip=flip and info))
    means=np.mean([c['costs'] for c in contexts],axis=0);fixed=int(np.argmin(means));oracle=float(np.mean([min(c['costs']) for c in contexts]));headroom=float(means[fixed]-oracle)
    comparator,geometry_mean=geometry_comparator(contexts);geometry_headroom=geometry_mean-oracle
    groups={(p['orientation'],p['surface'],p['distance']) for p in pairs if p['counted_flip']}
    gates=dict(complete_roster=len(contexts)==96,controllability=sum(c['controllable'] for c in contexts)>=72,informative_choice_groups=len(groups)>=6,oracle_headroom=geometry_headroom>=.1 and geometry_headroom>=.2*geometry_mean)
    return dict(geometry_target_comparator=comparator,geometry_target_mean_loss=geometry_mean,geometry_target_oracle_headroom=geometry_headroom,contexts=contexts,pairs=pairs,fixed_action_means=means.tolist(),strongest_fixed_action=fixed,oracle_mean_loss=oracle,oracle_headroom=headroom,controllable_contexts=sum(c['controllable'] for c in contexts),informative_choice_groups=len(groups),gates=gates,qualified=all(gates.values()))
