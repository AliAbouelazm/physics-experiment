"""Bounded state-dynamics correction. Native coverage and public geometry only."""
import copy
import math
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import torch
from src import pusht_task as v1

SURFACES=[(-60.,15.,-1.),(60.,15.,1.),(-15.,105.,-1.),(15.,105.,1.)]
POLYGONS=[np.array([[-60,0],[60,0],[60,30],[-60,30]]),np.array([[-15,30],[15,30],[15,120],[-15,120]])]
NAMES=['hold','up','right','left','down','up right','up left','down right','down left']+[f'{side} push {depth}' for side in ('left crossbar','right crossbar','left stem','right stem') for depth in (20,40)]


def shape(pose):
    x,y,angle=np.asarray(pose,dtype=float)
    c,s=math.cos(angle),math.sin(angle);rotation=np.array([[c,-s],[s,c]])
    return unary_union([Polygon(points@rotation.T+[x,y]) for points in POLYGONS])

GOAL=shape([256,256,math.pi/4])


def coverage(encoded):
    data=np.asarray(encoded,dtype=float)
    v1.finite(data,'coverage state')
    return float(shape([data[2]*512,data[3]*512,math.atan2(data[4],data[5])]).intersection(GOAL).area/GOAL.area)


def family(observed):
    observed=np.asarray(observed);result=[np.repeat(v1.target(observed,a)[None],16,axis=0) for a in v1.ACTIONS]
    angle=observed[4];c,s=math.cos(angle),math.sin(angle);rotation=np.array([[c,-s],[s,c]])
    for x,y,normal in SURFACES:
        for depth in (20,40):
            approach=np.clip(rotation@np.array([x+30*normal,y])+observed[2:4],30,482)
            push=np.clip(rotation@np.array([x-depth*normal,y])+observed[2:4],30,482)
            result.append(np.concatenate([np.repeat(approach[None],8,axis=0),np.repeat(push[None],8,axis=0)]))
    return np.stack(result)


def simulate(physics,state,seed,commands=None,candidate=None):
    if not 7000<=seed<9000:raise ValueError('Reserved or undeclared seed range')
    env=v1.make_env(physics);e=env.unwrapped
    try:
        obs,_=env.reset(seed=seed,options={'reset_to_state':list(state)})
        if candidate is not None:commands=family(obs)[candidate]
        commands=np.asarray(commands,dtype=float);observations=[obs.copy()];contacts=[];coverages=[];counter=[0]
        original=e.collision_handeler.post_solve
        def post(arbiter,space,data):
            original(arbiter,space,data);bodies=[s.body for s in arbiter.shapes]
            if any(b is e.agent for b in bodies) and any(b is e.block for b in bodies):counter[0]+=1
        e.collision_handeler.post_solve=post
        for command in commands:
            counter[0]=0;obs,_,_,_,info=env.step(command)
            observations.append(v1.finite(obs.copy(),'observation'));contacts.append(counter[0]>0);coverages.append(float(info['coverage']))
        return {'observations':np.asarray(observations),'commands':commands,'contacts':np.array(contacts),'coverage':np.array(coverages)}
    finally:env.close()


class Model(v1.Model):
    def __init__(self,scale=None):
        super().__init__();self.register_buffer('scale',torch.ones(6) if scale is None else scale.clone())
    def forward(self,state,previous,action):
        return super().forward(state,previous,action)*self.scale


def constrain(state,previous,next_state):
    """Only resting, geometrically separated blocks are constrained."""
    if bool(((state[2:]-previous[2:]).abs()<=1e-8).all()):
        a=state.detach().numpy();b=next_state.detach().numpy()
        poly=shape([a[2]*512,a[3]*512,math.atan2(a[4],a[5])])
        segment=LineString([a[:2]*512,b[:2]*512])
        if segment.distance(poly)>16:
            return torch.cat([next_state[:2],state[2:]])
    return next_state


def step(model,state,previous,action):
    nxt=v1.finite(state+model(state,previous,action),'v2 predicted state')
    rotation=nxt[4:]/nxt[4:].norm().clamp_min(1e-6)
    return constrain(state,previous,torch.cat([nxt[:4],rotation]))


def path(model,observed,commands,previous_observed=None):
    s=torch.tensor(v1.encode(observed));previous=s.clone() if previous_observed is None else torch.tensor(v1.encode(previous_observed));result=[s]
    with torch.no_grad():
        for command in commands:
            nxt=step(model,s,previous,torch.tensor(command,dtype=torch.float32));previous,s=s,nxt;result.append(s)
    return torch.stack(result)


def choose(models,observed):
    commands=family(observed)
    paths=torch.stack([torch.stack([path(m,observed,c) for c in commands]) for m in models])
    values=np.array([[coverage(p[-1]) for p in member] for member in paths]);mean=values.mean(0)
    return int(mean.argmax()),paths,mean


def score(models,observed):
    commands=[np.repeat(v1.target(observed,p)[None],8,axis=0) for p in v1.PROBES]
    paths=torch.stack([torch.stack([path(m,observed,c)[1:,2:]/m.scale[2:] for m in models]) for c in commands])
    scores=paths.var(1,unbiased=False).mean((1,2));v1.finite(scores,'v2 uncertainty')
    return {'scores':scores.tolist(),'selected_probe':int(scores.argmax())}


def adapt(models,observations,commands):
    copied=copy.deepcopy(models);batch=v1.batch_from_path(observations,commands);updates=[]
    for model in copied:
        before=model.adapter.weight.detach().clone();opt=torch.optim.SGD(model.adapter.parameters(),lr=.05);maximum=0.
        for _ in range(40):
            opt.zero_grad();loss=(((model(*batch[:3])-batch[3])/model.scale)**2).mean();v1.finite(loss,'v2 adapter loss');loss.backward()
            maximum=max(maximum,float(v1.finite(model.adapter.weight.grad.norm(),'v2 adapter gradient')));opt.step();v1.finite(model.adapter.weight,'v2 adapter weights')
        updates.append({'gradient_norm':maximum,'weight_change':float((model.adapter.weight.detach()-before).norm()),'loss':float(loss.detach())})
    return copied,updates


def data():
    batches=[];masks=[];metadata=[];seed=7200
    for physics in v1.TRAIN:
        for scene_id,scene in enumerate(v1.SCENES):
            for candidate in range(17):
                rng=np.random.default_rng(seed);initial=np.array(scene)
                initial[2:4]+=rng.uniform(-8,8,2);initial[4]+=rng.uniform(-.1,.1);initial[:2]+=rng.uniform(-5,5,2)
                r=simulate(physics,initial,seed,candidate=candidate);batch=v1.batch_from_path(r['observations'],r['commands']);batches.append(batch);masks.append(torch.tensor(r['contacts']))
                moving=(batch[0][:,2:]-batch[1][:,2:]).abs().max(1).values>1e-8
                metadata.append({'seed':seed,'physics':physics,'scene':scene_id,'candidate':candidate,'contact':int(r['contacts'].sum()),'moving_noncontact':int((moving&~masks[-1]).sum()),'stationary_noncontact':int((~moving&~masks[-1]).sum())});seed+=1
    return tuple(torch.cat([b[i] for b in batches]) for i in range(4)),torch.cat(masks),metadata


def train(batch,contact):
    scale=batch[3].square().mean(0).sqrt().clamp_min(.001);yes=torch.where(contact)[0];no=torch.where(~contact)[0]
    if not len(yes) or not len(no):raise ValueError('Missing contact sampling stratum')
    models=[];records=[]
    for seed in (1701,1702,1703):
        torch.manual_seed(seed);model=Model(scale);model.adapter.requires_grad_(False);before=torch.cat([p.detach().flatten() for p in model.base.parameters()]);opt=torch.optim.Adam(model.base.parameters(),lr=.003);rng=torch.Generator().manual_seed(seed);maximum=0.
        for _ in range(1200):
            ids=torch.cat([yes[torch.randint(len(yes),(64,),generator=rng)],no[torch.randint(len(no),(64,),generator=rng)]])
            b=[x[ids] for x in batch];opt.zero_grad();loss=(((model(*b[:3])-b[3])/scale)**2).mean();v1.finite(loss,'v2 training loss');loss.backward()
            maximum=max(maximum,float(v1.finite(torch.stack([p.grad.norm() for p in model.base.parameters()]).norm(),'v2 gradient')));opt.step()
        after=torch.cat([p.detach().flatten() for p in model.base.parameters()]);v1.finite(after,'v2 weights');model.base.requires_grad_(False);model.adapter.requires_grad_(True)
        for p in model.parameters():p.grad=None
        records.append({'seed':seed,'loss':float(loss.detach()),'gradient_norm':maximum,'weight_change':float((after-before).norm())});models.append(model)
    return models,records


def load(pathname):
    models=[]
    for state in torch.load(pathname,map_location='cpu',weights_only=True):
        model=Model();model.load_state_dict(state);model.base.requires_grad_(False);models.append(model)
    return models
