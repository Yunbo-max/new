"""R012 six-node CPU diagnostics; no pretrained model/GPU/formal review."""
import json, math, platform
import numpy as np
import torch
from scipy.special import softmax, hyp0f1
from scipy.optimize import root
from scipy.stats import t as student_t
seed=20261024
rng=np.random.default_rng(seed)
out={'environment':{'seed':seed,'numpy':np.__version__,'torch':torch.__version__,'python':platform.python_version()},'status':'synthetic CPU only'}
# A parent: asymmetric target-covariance confidence interval leads to an ordinary robust shrinker.
def robust(a,v,lo,hi):return -(a-np.clip(a,lo,hi))/v
v,a=2.,.75
assert abs(robust(a,v,-.4,.4)+.175)<1e-12
A={'known_a':a,'known_variance_D':v,'interval_examples':{},'synthetic_label_ci':{}}
for lo,hi in [(-.4,.4),(.85,1.05),(.5,.9)]:
    beta=robust(a,v,lo,hi)
    delta=lambda c:v*beta**2+2*beta*(a-c)
    A['interval_examples'][str((lo,hi))]={'beta':beta,'worst_excess_risk':max(delta(lo),delta(hi))}
    assert max(delta(lo),delta(hi))<=1e-12
for n in (8,32,128):
    cases={}
    for ctrue in (-.2,.2,.8,1.1):
        ne=3000
        D=rng.normal(scale=np.sqrt(v),size=(ne,n))
        theta=(ctrue/v)*D+rng.normal(scale=.6,size=(ne,n))
        ss=(D*D).sum(1)
        slope=(theta*D).sum(1)/ss
        residual=theta-slope[:,None]*D
        sigma2=(residual*residual).sum(1)/(n-1)
        radius=v*student_t.ppf(.975,n-1)*np.sqrt(sigma2/ss)
        lo=v*slope-radius;hi=v*slope+radius
        bet=robust(a,v,lo,hi)
        risk=v*bet*bet+2*bet*(a-ctrue)
        covered=(lo<=ctrue)&(ctrue<=hi)
        assert np.all(risk[covered]<1e-10)
        cases[str(ctrue)]={'coverage':float(covered.mean()),'nonzero_rate':float(np.mean(bet!=0)),
                           'mean_population_risk_delta':float(np.mean(risk)),
                           'positive_delta_and_CI_covers':int(np.sum((risk>1e-10)&covered))}
    A['synthetic_label_ci'][str(n)]=cases
out['QKV-A']=A
# A child: domain-level label-polarity inversion is fixed by known mixture likelihood.
g=.25;T=9.;sig2=1+T*.5/(T+.5);rho=.3
AA={'rho':rho,'g':g,'noise_variance':sig2,'no_label_excess':2*g*g,'n':{}}
for n in (4,16,64,256):
    episodes=200000
    s=rng.choice([-1.,1.],size=episodes)
    flip=rng.random(episodes)<rho
    S=2*rng.chisquare(n,size=episodes)
    stat=s*(1-2*flip.astype(int))*g*S+np.sqrt(sig2*S)*rng.normal(size=episodes)
    mu_nom=np.tanh((g/sig2)*stat)
    mu_aw=(1-2*rho)*mu_nom
    naive=2*g*g*np.mean((s-mu_nom)**2)
    corrected=2*g*g*np.mean((s-mu_aw)**2)
    AA['n'][str(n)]={'misspecified_bayes_excess':float(naive),'mixture_bayes_excess':float(corrected)}
    assert corrected<2*g*g+.002
AA['asymptotic_misspecified_excess']=8*g*g*rho
AA['asymptotic_mixture_excess']=8*g*g*rho*(1-rho)
out['QKV-A-signal-noise-identifiability']=AA
# B parent: two heads sharing one KV group, ordinary row temperatures can match a 2D
# radial group's output exactly even though each individual routing distribution differs.
s1=np.array([1.,0.,-1.]);s2=np.array([-.6,.2,.8])
rad=np.array([2.,1.,.7]);lograd=np.log(rad);eta=.15
V1=np.array([[1.,0.],[0.,1.],[-1.,0.]])
V2=np.array([[0.,1.],[1.,0.],[0.,-1.]])
heads=[(s1,V1),(s2,V2)]
def total(h):return sum(softmax(s)@V for s,V in h)
base=total(heads)
changed=total([(s*np.exp(eta*lograd),V) for s,V in heads])
def with_temps(alpha):return total([(alpha[0]*s1,V1),(alpha[1]*s2,V2)])
fit=root(lambda al:with_temps(al)-changed,[1.,1.])
assert fit.success and np.all(fit.x>0)
assert np.linalg.norm(with_temps(fit.x)-changed)<1e-9
individual_kl=[]
for j,(s,V) in enumerate(heads):
    p=softmax(s*np.exp(eta*lograd));q=softmax(fit.x[j]*s)
    individual_kl.append(float(np.sum(p*np.log(p/q))))
G=np.column_stack([sum(softmax(s)[k]*(V[k]-softmax(s)@V)*s[k] for k in range(3)) for s,V in heads])
B={'eta':eta,'group_output_change_norm':float(np.linalg.norm(changed-base)),
   'head_temperature_repair':fit.x.tolist(),'group_output_remaining_error':float(np.linalg.norm(with_temps(fit.x)-changed)),
   'each_head_routing_KL_remains_positive':individual_kl,'temperature_group_tangent_rank':int(np.linalg.matrix_rank(G)),
   'temperature_group_tangent_condition':float(np.linalg.cond(G))}
assert min(individual_kl)>1e-8
out['QKV-B']=B
# B child: collision of bf16 K does NOT imply loss of all state information if V carries radius.
eps=1e-6
def qnorm(x):
    xx=x.float();return (xx*torch.rsqrt(xx.square().mean(-1,keepdim=True)+eps)).to(x.dtype)
x=torch.ones(128,dtype=torch.bfloat16)
key1=qnorm(x);key2=qnorm(2*x)
assert torch.equal(key1,key2)
# Both keys match, attention remains 50/50; an independent V projection can carry old radius.
weights=softmax([0.,0.]);value_before=np.array([1.,0.]);value_after=np.array([2.,0.])
BB={'bitwise_equal_keys_x_vs_2x':True,'value_path_retains_original_radius_delta':float(weights@(value_after-value_before)),
    'when_values_equal_attention_output_delta':0.,
    'radial_log_sensitivity_eps_over_u_plus_eps':{str(k):1/(1+k) for k in (1,100,10000,1000000)},
    'radial_inverse_condition_number':{str(k):1+k for k in (1,100,10000,1000000)}}
assert abs(BB['value_path_retains_original_radius_delta']-.5)<1e-12
out['QKV-B-precision-radial-collision']=BB
# C parent: a planted aligned key's softmax phase threshold accounts for
# the background exp-logit moment; the iid isotropic rarity does not apply to it.
d=128;N=65536;sstar=math.sqrt(d)
mgf=float(hyp0f1(d/2,d/4))
approx=math.exp(sstar)/(math.exp(sstar)+N*mgf)
rep=64
normal=rng.normal(size=(rep,N));chi=rng.chisquare(d-1,size=(rep,N))
score=np.sqrt(d)*normal/np.sqrt(normal*normal+chi)
actual=np.exp(sstar)/(np.exp(sstar)+np.exp(score).sum(axis=1))
C={'d':d,'N_background':N,'log_partition_excess':math.log(mgf),
   'exact_isotropic_sphere_exp_score_moment':mgf,'perfect_key_score':sstar,
   'threshold_logN_plus_logMGF':math.log(N)+math.log(mgf),
   'analytic_mass_using_background_mgf':approx,
   'sampled_64_backgrounds_mass_mean':float(actual.mean()),
   'sampled_64_backgrounds_mass_std':float(actual.std(ddof=1)),
   'normal_score_mgf_approx_mass':math.exp(sstar)/(math.exp(sstar)+N*math.exp(.5))}
assert abs(actual.mean()-approx)<.002
out['QKV-C']=C
# C child: an ordinary fractional-shrink correction has a less restrictive
# certified improvement condition than the R009 all-or-nothing full correction.
def test_one(t,b):
    vmax=max(1.,abs(b));rmax=min(math.exp(t),1+.5*math.exp(t)*t*t)
    dw=2*rmax/(1+2*rmax)-2/3
    Ebound=.5*math.exp(t)*t*t*(2*vmax+t)
    Bound=2*vmax*dw+(2/3)*Ebound
    full=(b+2*math.sinh(t))/(1+2*math.cosh(t))
    proposed=(b+2*t)/3
    change=proposed-b;D=abs(change)
    fullcert=D>0 and Bound<D/2
    dampcert=D>0 and Bound<D
    lam=max(0.,1-Bound/D) if D>0 else 0
    damped=b+lam*change
    return fullcert,dampcert,lam,abs(full-b),abs(full-proposed),abs(full-damped),Bound,D
Nsim=20000
prng=np.random.default_rng(seed)
T=prng.uniform(0,2,size=Nsim);B=prng.normal(0,1,size=Nsim)
mat=np.array([test_one(float(t),float(b)) for t,b in zip(T,B)])
# Certified harmful correction would contradict the proved sufficient inequality.
assert not np.any((mat[:,1]>0)&(mat[:,5]>mat[:,3]+1e-10))
CC={'n':Nsim,'fixture':'evicted +/-t keys and +/-1 values, retained value Normal(0,1), t Uniform(0,2)',
    'full_correction_cert_fraction':float(mat[:,0].mean()),
    'damped_correction_cert_fraction':float(mat[:,1].mean()),
    'average_selected_damping_lambda':float(mat[:,2].mean()),
    'nonzero_lambda_median':float(np.median(mat[mat[:,2]>0,2])) if np.any(mat[:,2]>0) else 0.,
    'full_correction_true_improvement_fraction':float(np.mean(mat[:,4]<mat[:,3])),
    'damped_correction_true_improvement_fraction':float(np.mean(mat[:,5]<mat[:,3]-1e-14)),
    'false_certifications':int(np.sum((mat[:,1]>0)&(mat[:,5]>mat[:,3]+1e-10))),
    'proved_rule':'if norm(err_true_to_full_correction)<=B and D=norm(full_correction-plain)>B, lambda=1-B/D gives strictly improved squared error versus plain whenever D>0'}
assert CC['damped_correction_cert_fraction']>=CC['full_correction_cert_fraction']
out['QKV-C-unsafe-moment-correction']=CC
with open('/mnt/data/qkv_R012/R012_metrics.json','w') as f:
    json.dump(out,f,indent=2,ensure_ascii=False)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('synthetic_label_ci','n')} if isinstance(v,dict) else v for k,v in out.items() if k not in ('environment','status')},indent=2,ensure_ascii=False))
