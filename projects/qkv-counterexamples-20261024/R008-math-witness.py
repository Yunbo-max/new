"""R008 CPU analytic witness. No weights, datasets, GPU or network."""
import json, math
import numpy as np
import torch
from scipy.special import softmax
# A: full-pair, unlabeled calibration under two observationally identical task targets.
T,n,g=9.,24,1.
base=.5+(T+.5)/(n-3)
assert abs(base-20/21)<1e-12
assert abs((base+2*g*g)-62/21)<1e-12
# B: exact normalization source ordering, with unit learnable gain.
def qnorm(x):
    orig=x.dtype; y=x.float()
    return (y*torch.rsqrt(y.square().mean()+1e-6)).to(orig)
for dtype in [torch.float16,torch.bfloat16]:
    assert torch.equal(qnorm(torch.ones(128,dtype=dtype)),qnorm(torch.full((128,),2.,dtype=dtype)))
assert not torch.equal(qnorm(torch.ones(128)),qnorm(torch.full((128,),2.)))
# C: MomentKV Eq9/11/12 for one retained token M and two evicted +/-1 tokens.
def full(t,M):return (M+2*math.sinh(t))/(1+2*math.cosh(t))
def moment(t,M):return (M+2*t)/3
assert abs(moment(8,4)-full(8,4))>abs(4-full(8,4))
def evict(E,u,M):
    R=[(u,0.,M,'R')]+[(0.,float(k),float(v),str(j)) for j,(k,v) in enumerate(E)];lost=[]
    while len(R)>1:
        a=softmax([x[0] for x in R])
        if not lost:r=np.array([abs(x[2]) for x in R])
        else:
            km=np.mean([x[1] for x in lost]);vm=np.mean([x[2] for x in lost])
            c=sum(x[2]*(x[1]-km) for x in lost)/len(lost)
            r=np.array([abs(x[2]-vm-c*x[1]) for x in R])
        lost.append(R.pop(int(np.argmin(a*r))))
    return {x[3] for x in lost},R[0][3]
assert evict([(-8,-1),(8,1)],0,4)==({'0','1'},'R')
m=math.tanh(1)
assert evict([(-1,-1),(1,1)],5,m)==({'0','1'},'R')
assert abs(full(1,m)-m)<1e-12 and abs(moment(1,m)-full(1,m))>.15
worlds=[]
for s in [-1,1]:
    k=np.array([-1.,0.,1.]);v=k+s*4*np.array([1.,-2.,1.]);M=2.
    assert evict(list(zip(k,v)),10,M)==({'0','1','2'},'R')
    moments=[3,float(k.sum()),float(v.sum()),float((k*v).sum())]
    z=np.exp(np.r_[0.,k]);fullvalue=float(z@np.r_[M,v]/sum(z))
    worlds.append([moments,fullvalue,abs(fullvalue-M),abs(fullvalue-1.)])
assert worlds[0][0]==worlds[1][0]==[3,0.,0.,2.]
assert (worlds[0][2]-worlds[0][3])*(worlds[1][2]-worlds[1][3])<0
print(json.dumps({"A":base,"B":"FP16/BF16 collide, FP32 differs","C_large":[full(8,4),4,moment(8,4)],"C_small":[full(1,m),m,moment(1,m)],"C_same_moments":worlds},indent=2))
