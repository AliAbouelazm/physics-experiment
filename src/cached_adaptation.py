"""Tiny endpoint prior and two-probe response correction. No simulator access."""
import math
import numpy as np
import torch
from torch import nn
SEEDS=(9101,9102,9103)
ACTIONS=((0.,0),(.25,20),(.25,40),(.75,20),(.75,40),(1.5,20),(1.5,40))
PROBES=((.25,20),(1.,20))
FIT_CELLS=(0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29)
EVAL_CELLS=(2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31)

def features(actions):return np.array([[s/1.5,n/40,s*n/60] for s,n in actions],dtype=np.float64)
def dose(actions):return np.array([s*n/20 for s,n in actions],dtype=np.float64)
def ridge(observed,base):
    observed=np.asarray(observed,dtype=np.float64);base=np.asarray(base,dtype=np.float64)
    if observed.shape!=(2,3) or base.shape!=(2,3):raise ValueError('Exactly two three-component probe responses required')
    if not np.isfinite(observed).all() or not np.isfinite(base).all():raise ValueError('Nonfinite probe')
    phi=dose(PROBES);return (phi[:,None]*(observed-base)).sum(0)/(.1+np.sum(phi**2))

class Model(nn.Module):
    def __init__(self,seed):
        super().__init__();self.net=nn.Sequential(nn.Linear(3,16,dtype=torch.float64),nn.Tanh(),nn.Linear(16,3,dtype=torch.float64))
        gen=torch.Generator(device='cpu').manual_seed(seed)
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer,nn.Linear):
                    bound=1/math.sqrt(layer.in_features)
                    layer.weight.uniform_(-bound,bound,generator=gen);layer.bias.uniform_(-bound,bound,generator=gen)
    def forward(self,x):return self.net(x)

def fit(x,y,seed):
    x=torch.as_tensor(x,dtype=torch.float64);y=torch.as_tensor(y,dtype=torch.float64)
    if x.shape!=(128,3) or y.shape!=(128,3):raise ValueError('Whole-family fit requires 128 records')
    model=Model(seed);opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001);history=[]
    for step in range(300):
        opt.zero_grad();loss=(model(x)-y).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite fit')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();history.append(dict(step=step+1,loss=float(loss.detach()),unclipped_gradient_norm=float(norm)))
    model.eval();return model,history

def predict(model,probe_y):
    with torch.no_grad():
        base=model(torch.tensor(features(ACTIONS))).numpy();probe_base=model(torch.tensor(features(PROBES))).numpy()
    base[0]=0.;b=ridge(probe_y,probe_base);adapted=base+dose(ACTIONS)[:,None]*b;simple_b=ridge(probe_y,np.zeros((2,3)));simple=dose(ACTIONS)[:,None]*simple_b
    if not all(np.isfinite(x).all() for x in (base,adapted,simple)):raise ValueError('Nonfinite prediction')
    return dict(frozen=base,adapted=adapted,simple=simple),b,simple_b

def wrap(angle):return np.arctan2(np.sin(angle),np.cos(angle))
def losses(pred,distance):
    pred=np.asarray(pred);return (pred[:,0]-distance/20)**2+pred[:,1]**2+(2.5*wrap(.4*pred[:,2]))**2

def choices(pred):return {str(int(d)):int(np.argmin(losses(pred,d))) for d in (8.,20.,32.)}

def gates(frozen,adapted,simple,geometry_loss,orbit_losses):
    improved=sum(a<f and a<s for f,a,s in orbit_losses)
    return dict(prediction=adapted['prediction_mse']<=.95*frozen['prediction_mse'] and adapted['prediction_mse']<=simple['prediction_mse'],decision=frozen['mean_loss']-adapted['mean_loss']>=.01 and simple['mean_loss']-adapted['mean_loss']>=.01,orbits=improved>=3,geometry_aware=adapted['mean_loss']<=geometry_loss,improved_orbits=improved)
