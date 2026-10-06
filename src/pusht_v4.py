"""V4 route-cost correction. Public geometry and observations only."""
import heapq,math
import numpy as np
from shapely.geometry import Point,LineString,box
from shapely.ops import nearest_points
from src.pusht_v2 import shape,coverage,SURFACES
from src.pusht_task import encode
RADIUS=15.;MARGIN=3.;CAP=2.;BOUNDS=box(24,24,488,488)
CENTROID=np.array([0.,40.714285714285715]);RG2=2500.

def rotation(angle):return np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
def angle_difference(a,b):return math.atan2(math.sin(a-b),math.cos(a-b))
def motion(current,previous):
    if previous is None:return {'origin_pixels_per_second':0.,'angle_degrees_per_second':0.}
    return {'origin_pixels_per_second':float(np.linalg.norm(current[2:4]-previous[2:4])*10),'angle_degrees_per_second':abs(angle_difference(current[4],previous[4]))*180/math.pi*10}
def contact_geometry(observed,index):
    x,y,n=SURFACES[index];rot=rotation(observed[4]);return observed[2:4]+rot@np.array([x,y]),rot@np.array([n,0.])
def clear_segment(a,b,obstacle):
    line=LineString([a,b]);return BOUNDS.covers(line) and not line.intersects(obstacle)
def route(start,goal,pose):
    obstacle=shape(pose).buffer(18,join_style=2);start=np.asarray(start);goal=np.asarray(goal)
    if not BOUNDS.covers(Point(start)) or not BOUNDS.covers(Point(goal)) or obstacle.intersects(Point(start)) or obstacle.intersects(Point(goal)):return None
    if clear_segment(start,goal,obstacle):return [start.tolist(),goal.tolist()]
    vertices=list(shape(pose).buffer(19,join_style=2).exterior.coords)[:-1]
    nodes=[start.tolist(),goal.tolist()]+[list(v) for v in vertices if BOUNDS.covers(Point(v)) and not obstacle.intersects(Point(v))]
    graph=[[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i):
            if clear_segment(nodes[i],nodes[j],obstacle):
                cost=float(np.linalg.norm(np.array(nodes[i])-nodes[j]));graph[i].append((j,cost));graph[j].append((i,cost))
    pending=[(0.,0)];dist={0:0.};parent={}
    while pending:
        cost,node=heapq.heappop(pending)
        if cost>dist[node]:continue
        if node==1:
            path=[1]
            while path[-1]!=0:path.append(parent[path[-1]])
            return [nodes[i] for i in path[::-1]]
        for nxt,weight in graph[node]:
            value=cost+weight
            if value<dist.get(nxt,float('inf')):dist[nxt]=value;parent[nxt]=node;heapq.heappush(pending,(value,nxt))
    return None

def cap_target(current,target):
    delta=np.asarray(target)-current;length=np.linalg.norm(delta)
    return np.asarray(current)+delta*min(1.,CAP/max(length,1e-12))

def scores(observed):
    center=observed[2:4]+rotation(observed[4])@CENTROID;initial=coverage(encode(observed));result=[]
    for i in range(4):
        point,normal=contact_geometry(observed,i);force=-normal;r=point-center;h=r[0]*force[1]-r[1]*force[0]
        theta=observed[4]+2*h/(RG2+h*h);newcenter=center+2*force/(1+h*h/RG2);pred=observed.copy();pred[2:4]=newcenter-rotation(theta)@CENTROID;pred[4]=theta
        result.append(coverage(encode(pred))-initial)
    return result

class Servo:
    def __init__(self):
        self.phase='initial';self.surface=None;self.steps=0;self.phase_steps=0;self.push_steps=0;self.initial_coverage=None;self.failure=None;self.after_withdraw='approach'
    def fail(self,reason):self.failure=reason;self.phase='failed'
    def transition(self,phase):self.phase=phase;self.phase_steps=0
    def select(self,obs):
        values=scores(obs);choices=[]
        for i,value in enumerate(values):
            if value<=0:continue
            point,normal=contact_geometry(obs,i);path=route(obs[:2],point+normal*19,obs[2:])
            if path is None:continue
            estimate=route_controls(path)
            if estimate>240-self.phase_steps or estimate+80>400-self.steps:continue
            choices.append((-value/max(estimate,1),i,path))
        if not choices:return None,None
        _,i,path=min(choices,key=lambda item:(item[0],item[1]))
        return i,path
    def command(self,observed,previous=None):
        obs=np.asarray(observed,dtype=float);prev=None if previous is None else np.asarray(previous,dtype=float)
        if obs.shape!=(5,) or not np.isfinite(obs).all():raise ValueError('Invalid observation')
        details={'motion':motion(obs,prev),'route':None}
        if self.phase=='initial':
            self.initial_coverage=coverage(encode(obs))
            if not BOUNDS.covers(Point(obs[:2])):self.fail('invalid_initial_wall_geometry')
            elif shape(obs[2:]).distance(Point(obs[:2]))<18:self.fail('invalid_initial_overlap_or_margin')
            else:self.transition('approach')
        if self.steps>=400 and self.phase not in ('done','failed'):self.fail('active_budget_exhausted')
        if self.phase not in ('done','failed','withdraw') and coverage(encode(obs))>=self.initial_coverage+.06:
            self.after_withdraw='done';self.transition('withdraw')
        target=obs[:2].copy()
        if self.phase=='approach':
            if self.phase_steps>=240:self.fail('approach_budget_exhausted')
            else:
                if self.surface is None:self.surface,_=self.select(obs)
                if self.surface is None:self.fail('no_improving_reachable_surface')
                else:
                    point,normal=contact_geometry(obs,self.surface);goal=point+19*normal;path=route(obs[:2],goal,obs[2:]);details['route']=path
                    if path is None:self.fail('approach_route_unreachable')
                    elif np.linalg.norm(obs[:2]-goal)<=1.25:self.transition('push');self.push_steps=0
                    else:target=cap_target(obs[:2],np.array(path[1]))
        if self.phase=='push':
            point,normal=contact_geometry(obs,self.surface);difference=obs[:2]-point;normal_gap=float(difference@normal);tangent=float(abs(difference@np.array([-normal[1],normal[0]])))
            details.update(normal_gap=normal_gap,tangent_error=tangent)
            if normal_gap>23 or tangent>8:
                self.after_withdraw='approach';self.transition('withdraw')
            else:
                target=cap_target(obs[:2],point+14*normal);self.push_steps+=1
                if self.push_steps>=4:
                    preferred=int(np.argmax(scores(obs)));self.push_steps=0
                    if preferred!=self.surface:self.after_withdraw='approach';self.transition('withdraw')
        if self.phase=='withdraw':
            poly=shape(obs[2:]);point=np.array(nearest_points(poly,Point(obs[:2]))[0].coords[0]);out=obs[:2]-point;distance=float(np.linalg.norm(out))
            if self.phase_steps>=40:self.fail('withdrawal_budget_exhausted')
            elif distance>=22:
                if self.after_withdraw=='done':self.transition('done')
                else:self.surface=None;self.transition('approach')
            elif distance<14.5:self.fail('invalid_withdrawal_overlap')
            else:
                target=cap_target(obs[:2],point+out/distance*23)
                if not clear_segment(obs[:2],target,poly.buffer(14.49,join_style=2)):self.fail('withdrawal_route_unreachable')
        if self.phase in ('failed','done'):target=obs[:2].copy()
        if self.phase not in ('done','failed'):
            self.steps+=1;self.phase_steps+=1
        assert np.linalg.norm(target-obs[:2])<=CAP+1e-9
        details.update(phase=self.phase,surface=self.surface,failure=self.failure)
        return target,details


def tracker_first_advance(cap=CAP):
    # Public PD gains and integration, no environment or hidden block physics.
    position=0.;velocity=0.
    for _ in range(10):
        velocity+=(100*(cap-position)-20*velocity)*.01
        position+=velocity*.01
    return position

def route_controls(path):
    total=12+4*max(0,len(path)-2)
    for a,b in zip(path,path[1:]):
        length=float(np.linalg.norm(np.array(a)-b));position=0.;velocity=0.;controls=0
        while position<length-1.25:
            target=min(length,position+CAP)
            for _ in range(10):
                velocity+=(100*(target-position)-20*velocity)*.01
                position+=velocity*.01
            controls+=1
            if controls>2000:return 1000000
        total+=controls
    return total

def observed_twist(obs,previous):
    if previous is None:return np.zeros(2),0.
    center=obs[2:4]+rotation(obs[4])@CENTROID
    before=previous[2:4]+rotation(previous[4])@CENTROID
    return (center-before)*10,angle_difference(obs[4],previous[4])*10

class Brake:
    """Bounded intercept using observed surface velocity, never hidden state."""
    def __init__(self):
        self.steps=0;self.quiet=0;self.surface=None;self.failure=None
    def command(self,observed,previous=None):
        obs=np.asarray(observed,dtype=float);prev=None if previous is None else np.asarray(previous,dtype=float)
        if obs.shape!=(5,) or not np.isfinite(obs).all():raise ValueError('Invalid observation')
        velocity,omega=observed_twist(obs,prev);measured=motion(obs,prev)
        details={'phase':'observe','surface':self.surface,'route':None,'motion':measured,'failure':self.failure}
        target=obs[:2].copy()
        if self.steps>=120:self.failure='brake_budget_exhausted'
        if prev is not None and measured['origin_pixels_per_second']<=1 and measured['angle_degrees_per_second']<=1:self.quiet+=1
        else:self.quiet=0
        if self.failure:details.update(phase='failed',failure=self.failure);return target,details
        if self.quiet>=10:details['phase']='done';return target,details
        self.steps+=1
        if prev is None:return target,details
        poly=shape(obs[2:]);nearest=np.array(nearest_points(poly,Point(obs[:2]))[0].coords[0]);separation=np.linalg.norm(obs[:2]-nearest)
        if separation<14.5:self.failure='invalid_brake_overlap'
        elif separation<19 and self.surface is None:
            target=cap_target(obs[:2],nearest+(obs[:2]-nearest)/separation*23);details['phase']='withdraw'
            if not clear_segment(obs[:2],target,poly.buffer(14.49,join_style=2)):self.failure='brake_withdrawal_unreachable'
        else:
            center=obs[2:4]+rotation(obs[4])@CENTROID;choices=[]
            for i in range(4):
                point,normal=contact_geometry(obs,i);lever=point-center
                surface_velocity=velocity+omega*np.array([-lever[1],lever[0]])
                closing=float(normal@surface_velocity)
                if closing<=.05:continue
                # Half-second visible constant-twist lead, clipped to five pixels.
                lead=surface_velocity*.5;lead*=min(1.,5/max(np.linalg.norm(lead),1e-12))
                goal=point+normal*19+lead;path=route(obs[:2],goal,obs[2:])
                difference=obs[:2]-point;tangent=abs(difference@np.array([-normal[1],normal[0]]));gap=difference@normal
                near=14.5<=gap<=23 and tangent<=6
                if near:choices.append((-closing,i,None,point,normal,True))
                elif path is not None and route_controls(path)+12<=120-self.steps:choices.append((-closing/max(route_controls(path),1),i,path,point,normal,False))
            if not choices:
                self.surface=None;details['phase']='observe_no_feasible_intercept'
            else:
                _,self.surface,path,point,normal,near=min(choices,key=lambda x:(x[0],x[1]));details.update(surface=self.surface,route=path)
                if near:
                    # Stay just outside the observed boundary to oppose outward motion.
                    target=cap_target(obs[:2],point+15*normal);details['phase']='brake'
                else:target=cap_target(obs[:2],np.array(path[1]));details['phase']='intercept'
        if self.failure:target=obs[:2].copy();details.update(phase='failed',failure=self.failure)
        assert np.linalg.norm(target-obs[:2])<=CAP+1e-9
        return target,details
