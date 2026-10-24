"""R014 compact deterministic CPU witnesses, no pretrained model run."""
import math,json
import numpy as np,torch
from scipy.special import logsumexp
rng=np.random.default_rng(20261024)
# A parent: correlated head errors; clear finite covariance harm threshold
rho=-.8;z=rng.standard_normal((120000,2))
e1=z[:,0];e2=rho*z[:,0]+math.sqrt(1-rho*rho)*z[:,1]
assert np.mean(e2**2)>np.mean((e1+e2)**2)
# A child: closed-form worst-case robust Bayes sign shrinkage
g=.25;robust=max(0,1-2*.45);nominal=1-2*.30
regret=lambda k,r:2*g*g*(1+k*k-2*k*(1-2*r))
assert regret(robust,.45)<.125<regret(nominal,.45)
# B parent: isotropic null projection expects 1-H/D squared residual
D=1024
frac={}
for H in (2,16):
    G=rng.standard_normal((D,H));Q,_=np.linalg.qr(G)
    V=rng.standard_normal((300,D))
    p=V@Q@Q.T
    frac[H]=float(np.sum((V-p)**2)/np.sum(V**2))
    assert abs(frac[H]-(D-H)/D)<.006
# B child: normalized BF16 K collision can coexist with distinct V
x=torch.ones(128,dtype=torch.bfloat16)
norm=lambda v:(v.float()*torch.rsqrt(v.float().square().mean()+1e-6)).to(v.dtype)
assert torch.equal(norm(x),norm(2*x))
assert float(x.float().mean())!=float((2*x).float().mean())
# C parent: value contrast, not target attention mass alone
N=65536;d=128;score=math.sqrt(d)
z=rng.standard_normal(N);s=score*z/np.sqrt(z*z+rng.chisquare(df=d-1,size=N))
lnB=float(logsumexp(s));pstar=float(np.exp(score-np.logaddexp(score,lnB)))
assert .40<pstar<.46
# C child Eq35: Jensen estimate b=1; exact evicted=2, retained=1
a=b=eps=1.
wR=a/(a+(1+eps)*b);hatR=a/(a+b);hatE=b/(a+b)
actual=hatR-wR
printed=wR*hatE*eps/(1+eps*hatE)
assert abs(actual-1/6)<1e-12 and abs(printed-1/9)<1e-12
assert abs(eps*wR*hatE-actual)<1e-12
for order in (1,2,3):
    k=np.arange(order+2,dtype=float)
    delta=np.array([(-1)**i*math.comb(order+1,i) for i in range(order+2)])
    assert np.allclose([delta@(k**r) for r in range(order+1)],0)
    assert abs(delta@np.exp(k)-(1-math.e)**(order+1))<1e-9
out={"A_theory_original":2+2*rho,"A_theory_after":1.,
"A_measured_original":float(np.mean((e1+e2)**2)),"A_measured_after":float(np.mean(e2**2)),
"A_child_rho45_nominal":regret(nominal,.45),"A_child_rho45_robust":regret(robust,.45),
"B_null_residual_fraction":frac,"B_K_collision_V_bypass":True,
"C_target_mass":pstar,"C_background_logZ":lnB,
"C_Eq35_gap_correct":actual,"C_Eq35_gap_printed":printed,
"C_moment_orders_checked":[1,2,3],"cpu_only":True}
print(json.dumps(out,indent=2))