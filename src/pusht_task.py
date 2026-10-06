"""Small learned state model on official PushT. Pixels are visualization only."""
import copy
import math
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import gymnasium as gym
import gym_pusht  # registers official environment
import numpy as np
import torch
from torch import nn

TRAIN = [(d, x) for d in (.2, .8) for x in (-15., 15.)]
DEVELOPMENT = [(d, x) for d in (.05, .95) for x in (-25., 25.)]
FINAL = [(d, x) for d in (.35, .65) for x in (-30., 30.)]
SCENES = [[256.,410.,256.,256.,0.], [170.,340.,256.,256.,0.], [342.,340.,256.,256.,0.]]
PROBES = np.array([[0,0],[0,-80],[80,0],[-80,0],[0,80]], dtype=np.float64)
ACTIONS = np.array([[0,0],[0,-120],[120,0],[-120,0],[0,120],[85,-85],[-85,-85],[85,85],[-85,85]], dtype=np.float64)


def finite(x, name):
    if not bool(torch.isfinite(torch.as_tensor(x)).all()): raise ValueError(f'Nonfinite {name}')
    return x


def encode(obs):
    obs = np.asarray(obs)
    return np.concatenate([obs[..., :4]/512, np.sin(obs[...,4:5]), np.cos(obs[...,4:5])], axis=-1).astype(np.float32)


def target(state, offset):
    return np.clip(np.asarray(state[:2])+offset, 30, 482)


def make_env(physics):
    if tuple(physics) in FINAL: raise ValueError('Reserved final physics may not be instantiated')
    return gym.make('gym_pusht/PushT-v0', obs_type='state', render_mode='rgb_array',
                    damping=physics[0], block_cog=(physics[1],45.),
                    visualization_width=512, visualization_height=512, disable_env_checker=True)


def simulate(physics, state, seed, commands, render=False):
    if not 7000 <= seed < 9000: raise ValueError('Only declared training/development seed range allowed')
    env = make_env(physics)
    try:
        obs, _ = env.reset(seed=seed, options={'reset_to_state': list(state)})
        observations = [finite(obs.copy(), 'observation')]; coverage=[]; frames=[]
        if render: frames.append(env.render())
        for command in commands:
            obs, _, _, _, info = env.step(np.asarray(command,dtype=np.float64))
            observations.append(finite(obs.copy(),'observation')); coverage.append(float(info['coverage']))
            if render: frames.append(env.render())
        return {'observations': np.stack(observations), 'coverage': coverage, 'frames': frames}
    finally: env.close()


def features(state, previous, action):
    return torch.cat([state, state-previous, action/512-state[..., :2]], dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(14,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,6))
        self.adapter = nn.Linear(14,6,bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, state, previous, action):
        x = features(state,previous,action)
        return finite(self.base(x)+self.adapter(x), 'predicted increment')


def model_path(model, observed, command, steps):
    s = torch.as_tensor(encode(observed)); previous=s.clone(); path=[s]
    action=torch.as_tensor(command,dtype=torch.float32)
    for _ in range(steps):
        nxt=finite(s+model(s,previous,action),'predicted state')
        rotation=nxt[4:]/nxt[4:].norm().clamp_min(1e-6)
        nxt=torch.cat([nxt[:4],rotation]);previous,s=s,nxt;path.append(s)
    return torch.stack(path)


def pose_cost(encoded):
    goal=torch.tensor([.5,.5],dtype=encoded.dtype,device=encoded.device)
    angle_alignment=(encoded[...,4]+encoded[...,5])/math.sqrt(2)
    return finite(((encoded[...,2:4]-goal)**2).sum(-1)+.1*(1-angle_alignment.clamp(-1,1)), 'pose cost')


def regret(costs, selected):
    finite(costs,'candidate costs')
    return float((costs[selected]-costs.min()).clamp_min(0))


def scores(ensemble, observed):
    with torch.no_grad():
        paths=torch.stack([torch.stack([model_path(m,observed,target(observed,p),8)[1:,2:] for m in ensemble]) for p in PROBES])
        uncertainty=finite(paths.var(1,unbiased=False).mean((1,2)), 'uncertainty')
    tied=torch.nonzero(uncertainty.max()-uncertainty<=1e-9).flatten().tolist()
    return {'uncertainty':uncertainty.tolist(),'selected_probe':tied[0],'tie_candidate_ids':tied}


def choose(ensemble, observed):
    with torch.no_grad():
        paths=torch.stack([torch.stack([model_path(m,observed,target(observed,a),12) for a in ACTIONS]) for m in ensemble])
        costs=pose_cost(paths[:,:,-1,:]).mean(0)
    return int(costs.argmin()),paths,costs


def batch_from_path(observations, commands):
    s=torch.tensor(encode(observations[:-1]));previous=torch.cat([s[:1],s[:-1]])
    action=torch.tensor(np.asarray(commands),dtype=torch.float32)
    delta=torch.tensor(encode(observations[1:]))-s
    return s,previous,action,delta


def fit_adapter(model, batch):
    before=model.adapter.weight.detach().clone();opt=torch.optim.SGD(model.adapter.parameters(),lr=.05)
    s,previous,action,y=batch;maximum=0.
    for _ in range(40):
        opt.zero_grad();loss=((model(s,previous,action)-y)**2).mean();finite(loss,'adapter loss')
        loss.backward();grad=float(finite(model.adapter.weight.grad.norm(),'adapter gradient'))
        maximum=max(maximum,grad);opt.step();finite(model.adapter.weight,'adapter weights')
    return {'loss':float(loss.detach()),'gradient_norm':maximum,
            'weight_change':float((model.adapter.weight.detach()-before).norm())}


def adapt(ensemble, observations, commands):
    copied=copy.deepcopy(ensemble);batch=batch_from_path(observations,commands)
    updates=[fit_adapter(m,batch) for m in copied]
    return copied,updates


def training_data():
    batches=[];metadata=[];seed=7100
    for physics in TRAIN:
        for scene_id,scene in enumerate(SCENES):
            for _ in range(4):
                rng=np.random.default_rng(seed);initial=np.array(scene)
                initial[2:4]+=rng.uniform(-12,12,2);initial[4]+=rng.uniform(-.15,.15)
                commands=[]
                for t in range(24):
                    if t%4==0: command=target(initial,ACTIONS[rng.integers(9)]+rng.uniform(-10,10,2))
                    commands.append(command.copy())
                result=simulate(physics,initial,seed,commands)
                batches.append(batch_from_path(result['observations'],commands))
                metadata.append({'seed':seed,'scene':scene_id,'physics':list(physics),'transitions':24})
                seed+=1
    return tuple(torch.cat([b[i] for b in batches]) for i in range(4)),metadata


def train(batch):
    ensemble=[];records=[]
    for seed in (701,702,703):
        torch.manual_seed(seed);m=Model();m.adapter.requires_grad_(False);before=torch.cat([x.detach().flatten() for x in m.base.parameters()])
        opt=torch.optim.Adam(m.base.parameters(),lr=.003);generator=torch.Generator().manual_seed(seed)
        maximum=0.
        for _ in range(600):
            indices=torch.randint(len(batch[0]),(128,),generator=generator)
            s,previous,action,y=(x[indices] for x in batch)
            opt.zero_grad();loss=((m(s,previous,action)-y)**2).mean();finite(loss,'base loss');loss.backward()
            maximum=max(maximum,float(torch.stack([x.grad.norm() for x in m.base.parameters()]).norm()))
            opt.step()
        m.base.requires_grad_(False);m.adapter.requires_grad_(True)
        for parameter in m.parameters(): parameter.grad=None
        after=torch.cat([x.detach().flatten() for x in m.base.parameters()]);finite(after,'trained base')
        records.append({'seed':seed,'loss':float(loss.detach()),'gradient_norm':maximum,'weight_change':float((after-before).norm())})
        ensemble.append(m)
    return ensemble,records


def load_models(path):
    weights=torch.load(path,map_location='cpu',weights_only=True)
    models=[]
    for state in weights:
        m=Model();m.load_state_dict(state);m.base.requires_grad_(False);models.append(m)
    return models
