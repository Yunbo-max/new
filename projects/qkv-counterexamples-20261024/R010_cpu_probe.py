"""R010 finite manual wake: six independent QKV mathematical diagnostic checks.

No pretrained weights, GPU, external APIs, formal reviewer, or native benchmark.
Python 3.13; numpy/scipy/torch CPU. Mathematical experimental fixtures only.
"""
import json, math, platform
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax, roots_genlaguerre, roots_hermitenorm, gammaln, betainc
from scipy.stats import t as student_t
from scipy.optimize import minimize_scalar,brentq

OUT={"environment":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"scipy":"1.17.0","torch_cuda":torch.cuda.is_available(),"seed":20261024}}

# A parent: robust calibration against unknown signal covariance |Cov(theta,D)|<=C.
# Observable a=Cov(y1,D), v=Var(D); objective change from beta=0 is
# v beta^2 + 2 beta (a-c) with true but inaccessible c=Cov(theta,D).
def delta_risk(beta,a,c,v): return v*beta*beta + 2*beta*(a-c)
def minimax_beta(a,C,v): return -math.copysign(max(abs(a)-C,0.),a)/v if a!=0 else 0.
a,C,v=.75,.4,2.
beta_proxy=-a/v
beta_safe=minimax_beta(a,C,v)
max_risk=lambda b:max(delta_risk(b,a,-C,v),delta_risk(b,a,C,v))
A_parent={"observable_cov":a,"signal_cov_envelope":C,"difference_variance":v,
          "naive_unlabeled_beta":beta_proxy,"minimax_beta":beta_safe,
          "naive_worst_excess_MSE":max_risk(beta_proxy),
          "minimax_worst_excess_MSE":max_risk(beta_safe),
          "do_nothing_worst_excess_MSE":max_risk(0.),
          "closed_form_minimax_worst_excess_MSE":-(max(abs(a)-C,0.)**2)/v,
          "robust_zero_c0.8":minimax_beta(a,.8,v)}
assert A_parent["naive_worst_excess_MSE"]>0
assert A_parent["minimax_worst_excess_MSE"]<0
assert abs(A_parent["minimax_worst_excess_MSE"]-A_parent["closed_form_minimax_worst_excess_MSE"])<1e-12
for x in np.linspace(-1,1,1001):
    assert max_risk(beta_safe)<max_risk(x)+1e-9
OUT["QKV-A"]=A_parent

# A child: unlike hard sign-classification, Bayes posterior-mean coefficient
# is NEVER worse than the unlabeled minimax midpoint in correctly specified model.
T,g,slabel_sq=9.,.25,1.
sigma2=slabel_sq+T*.5/(T+.5)
sigma=math.sqrt(sigma2)
# d~N(0,2), observed label residual t_i=s*g*d_i+zeta_i;
# LLR/2 = (g/sigma2)sum_i(d_i*t_i); posterior E[s|data]=tanh(LLR/2).
# Sum d_i^2=4*Gamma(n/2,1), and conditional A=(g/sigma2)*sum(d_i*t_i)
#      = (4g^2/sigma2)*G + (2g/sigma)*sqrt(G)*Z, Z~N(0,1).
# Bayes excess conditional on data is 2g^2*sech^2(A).
nodesZ,wZ=roots_hermitenorm(100)
wZ=wZ/math.sqrt(2*math.pi)
A_child={"task_sign_g":g,"noise_sigma2":sigma2,"no_label_excess_MSE":2*g*g,"n":{}}
for n in (1,4,16,64):
    alpha=n/2
    nodesG,wG=roots_genlaguerre(90,alpha-1)
    wG=wG/math.exp(float(gammaln(alpha)))
    A=(4*g*g/sigma2)*nodesG[:,None] + (2*g/sigma)*np.sqrt(nodesG[:,None])*nodesZ[None,:]
    sech2=4*np.exp(-2*np.abs(A))/(1+np.exp(-2*np.abs(A)))**2
    bayes_regret=float(2*g*g*np.sum(wG[:,None]*wZ[None,:]*sech2))
    pe=float(student_t.cdf(-g*math.sqrt(2*n)/sigma,df=n))
    hard_regret=8*g*g*pe
    A_child["n"][str(n)]={"hard_sign_error_probability":pe,"hard_sign_excess_MSE":hard_regret,
        "posterior_mean_excess_MSE":bayes_regret,
        "posterior_beats_midpoint":bayes_regret<2*g*g,
        "hard_sign_worse_than_midpoint":hard_regret>2*g*g}
assert A_child['n']['4']['hard_sign_worse_than_midpoint']
assert not A_child['n']['4']['posterior_mean_excess_MSE']>=2*g*g
assert A_child['n']['16']['posterior_mean_excess_MSE']<A_child['n']['4']['posterior_mean_excess_MSE']
OUT["QKV-A-signal-noise-identifiability"]=A_child

# B parent: positive-temperature can exactly emulate any nearby 2-token routing
# provided original score gap nonzero and required scaling alpha>0.
# With 3+ tokens shape differences survive temperature, but with nonconstant
# value feature V=s, KL-opt temperature matches value-output EXACTLY.
s2=np.array([1.,-1.]); a2=np.log([2.,.7]); eta=.3
ell2=s2*np.exp(eta*a2);alpha2=float((ell2[0]-ell2[1])/(s2[0]-s2[1]))
p2=softmax(ell2);q2=softmax(alpha2*s2)
assert alpha2>0 and np.max(np.abs(p2-q2))<1e-14
s3=np.array([1.,0.,-1.]);a3=np.log([2.,1.,.7]);ell3=s3*np.exp(eta*a3)
p3=softmax(ell3)
# minimize KL(p_eta||softmax(alpha*s))
kl=lambda alpha:float(np.sum(p3*np.log(p3/softmax(alpha*s3))))
res=minimize_scalar(kl,bounds=(0,3),method='bounded',options={'xatol':1e-13})
alpha_mom=brentq(lambda aa: float(softmax(aa*s3)@s3-p3@s3),0,3,xtol=1e-14)
q3=softmax(alpha_mom*s3)
# analytical temp optimality gives E_q s == E_p s, V=s is nonconstant.
V=s3
B_parent={"eta":eta,"two_token_exact_temperature":alpha2,"two_token_max_prob_error":float(np.max(abs(p2-q2))),
          "three_token_best_temperature":float(alpha_mom),"three_token_min_KL":float(res.fun),
          "three_token_V_equals_original_logits_output_error":float(abs((p3-q3)@V)),
          "three_token_distribution_l1_error":float(np.sum(abs(p3-q3)))}
assert B_parent['three_token_min_KL']>1e-6
assert B_parent['three_token_V_equals_original_logits_output_error']<1e-9
OUT['QKV-B']=B_parent

# B precision child: EXACT real-valued radial sensitivity vs output dtype resolution.
# F(a*x)/F(x)=a sqrt(u+eps)/sqrt(a*a*u+eps), u=||x||^2/d.
eps=1e-6
B_child={"eps":eps,"d":128,"u_over_eps_grid":{}}
for rr in (.01,.1,1,10,100,1e3,1e6):
    u=rr*eps
    exact=float(2*math.sqrt(u+eps)/math.sqrt(4*u+eps))
    row={"radial_ratio_output_exact":exact,"relative_scale_signal_exact":exact-1,
         "derivative_dlogF_dlogx":eps/(u+eps),"postcast_radial_collision":{}}
    for name,dtype in [('bf16',torch.bfloat16),('fp16',torch.float16),('fp32',torch.float32)]:
        x=torch.full((128,),math.sqrt(u),dtype=dtype)
        def norm(z):
            y=z.float()
            return (y*torch.rsqrt(y.square().mean(-1,keepdim=True)+eps)).to(z.dtype)
        p,q=norm(x),norm(2*x)
        row["postcast_radial_collision"][name]=bool(torch.equal(p,q))
    B_child['u_over_eps_grid'][str(rr)]=row
assert B_child['u_over_eps_grid']['1']['postcast_radial_collision']['bf16']==False
assert B_child['u_over_eps_grid']['1000000.0']['postcast_radial_collision']['bf16']==True
u=1.;c=2*math.sqrt(u+eps)/math.sqrt(4*u+eps)
# Single K with original logit score s=10, relative radial scale c-1,
# max variation in attention output with max value-norm Vmax is <=2*Vmax*tanh(|delta_s|/4).
B_child['single_key_output_change_bound_unit_V_s10']={"relative_scale_delta":c-1,
 "max_logit_shift_at_s10":10*(c-1),
 "max_output_shift_for_normV_le1":2*math.tanh(10*(c-1)/4)}
OUT['QKV-B-precision-radial-collision']=B_child

# C parent: head RMSNorm norm bound vs high logits; the geometric probability
# under *independent isotropic random* normalized directions is very small.
d=128
C_parent={"head_dim":d,"unit_gain_theoretical_abs_logit_ceiling":math.sqrt(d),
          "random_uniform_unit_sphere_query_key":{"description":"keys and queries independent uniformly isotropic, exactly normalized to sqrt(d)","N":{}}}
for N in [1024,4096,8192,65536]:
    t=math.log(N);cosv=t/math.sqrt(d)
    prob= float(betainc((d-1)/2,.5,max(0.,1-cosv*cosv))/2) if cosv<1 else 0.
    C_parent['random_uniform_unit_sphere_query_key']['N'][str(N)]={"lnN":t,"required_cosine":cosv,
      "prob_random_one_token_meets_lnN":prob,"union_bound_N_tokens":min(1.,N*prob),
      "feasible_under_unit_gain_norm_bound":cosv<=1}
assert C_parent['random_uniform_unit_sphere_query_key']['N']['65536']['feasible_under_unit_gain_norm_bound']
assert C_parent['random_uniform_unit_sphere_query_key']['N']['65536']['union_bound_N_tokens']<1e-50
OUT['QKV-C']=C_parent

# C child: counterexample survives ARBITRARILY SMALL query spread at the operator level.
# The retained token has output M=tanh(t), equal to true evicted output; plain eviction exact.
# MomentKV Eq 9,11,12 predicts (M+2t)/3, so harm=(2/3)(t-tanh(t))=2t^3/9+O(t^5).
C_child={"zero_eviction_error_any_positive_t":True,"small_spread":{},"synthetic_certificate":{}}
for t in (.01,.05,.1,.25,1,8):
    m=math.tanh(t)
    full=(m+2*math.sinh(t))/(1+2*math.cosh(t))
    corrected=(m+2*t)/3
    err=abs(corrected-full)
    C_child['small_spread'][str(t)]={"t":t,"retained_M":m,"plain_eviction_error":abs(m-full),
       "corrected_output_abs_error":err,
       "asymptotic_error_2t3over9":2*t**3/9,
       "scaled_error_div_t_cubed":err/(t**3)}
    assert abs(m-full)<1e-12 and err>0
# Algorithm 1 reachable partition for every tested t>0 (unbounded-score toy).
# In 2D: previous query sees old logits (L,0,0), later query (0,+t,-t).
# Retain (key_old=L, value=tanh(t)) and evict E pair (old_key0, values +1,-1).
# When the first E has been removed, its centered moment is zero; residual v-v_E.
# If exp(L)*M>1 and exp(L)*(1-M)>2, both E are evicted before R.
def momentkv_old_query_selection(t):
    M=math.tanh(t)
    L=math.log(max(1/M,2/(1-M)))+1
    alive=[('R',L,M),('E+',0.,1.),('E-',0.,-1.)]
    lost=[]
    while len(alive)>1:
        w=softmax([item[1] for item in alive])
        if not lost:
            residual=[abs(item[2]) for item in alive]
        else:
            assert len(lost)==1
            residual=[abs(item[2]-lost[0][2]) for item in alive]
        idx=int(np.argmin(w*np.asarray(residual)))
        lost.append(alive.pop(idx))
    return {'t':t,'old_retained_logit':L,'selected_eviction_order':[x[0] for x in lost],
            'survivor':alive[0][0],'sufficient_conditions':{
            'expL_times_M':math.exp(L)*M,'expL_times_1minusM':math.exp(L)*(1-M)}}
C_child['old_query_algorithm1_eviction_traces']={}
for t in (.01,.05,.1,1.):
    check=momentkv_old_query_selection(t)
    assert check['survivor']=='R' and set(check['selected_eviction_order'])=={'E+','E-'}
    assert check['sufficient_conditions']['expL_times_M']>1
    assert check['sufficient_conditions']['expL_times_1minusM']>2
    C_child['old_query_algorithm1_eviction_traces'][str(t)]=check

# conservative envelope certificate from R009. Simulated within a precisely defined
# scalar, two-evicted-key distribution. No native model.
rng=np.random.default_rng(20261024)
def simple_fixture(t,b):
    vmax=max(1.,abs(b))
    abs_t=abs(t)
    if abs_t>15: return False,False # regime not used
    rmax=min(math.exp(abs_t),1+.5*math.exp(abs_t)*t*t)
    what=2/3
    wmax=2*rmax/(1+2*rmax)
    dw=wmax-what
    BE=.5*math.exp(abs_t)*t*t*(2*vmax+abs(t))
    BB=2*vmax*dw+what*BE
    full=(b+2*math.sinh(t))/(1+2*math.cosh(t))
    corrected=(b+2*t)/3
    predmove=abs(corrected-b)
    cert=(predmove>0 and BB<predmove/2)
    better=abs(corrected-full)<abs(b-full)
    return cert,better
for cap in (.1,.5,1.,2.):
    nn=20000
    ts=rng.uniform(0,cap,nn)
    bs=rng.normal(0,1,nn)
    status=np.array([simple_fixture(float(t),float(b)) for t,b in zip(ts,bs)],dtype=bool)
    cert= status[:,0]
    helps=status[:,1]
    C_child['synthetic_certificate'][str(cap)]={"draws":nn,"distribution":f"t~Uniform(0,{cap}), b~N(0,1), E keys +t/-t and values +1/-1",
       "cert_pass_fraction":float(np.mean(cert)),"correction_helps_fraction":float(np.mean(helps)),
       "cert_false_positive_count":int(np.sum(cert & ~helps)),"certified_examples":int(np.sum(cert))}
    assert np.sum(cert & ~helps)==0
# Memory all payloads charged at fixed dk=dv=128.
dk=dv=128
mem_fp32=dk*dv*4+(dk+dv)*4+8+2*4
mem_fp16=dk*dv*2+(dk+dv)*2+8+2*4
entry=(dk+dv)*2+8
C_child['memory_bytes_per_KV_head']={"FP32_dense_moment_means_count_two_envelopes":mem_fp32,
 "BF16_dense_moment_means_count_FP32_two_envelopes":mem_fp16,
 "BF16_exact_one_KV_plus_int64_position":entry,
 "FP32_moment_exact_entry_equivalent":mem_fp32/entry,
 "BF16_moment_exact_entry_equivalent":mem_fp16/entry}
OUT['QKV-C-unsafe-moment-correction']=C_child

p=Path(__file__).with_name('R010_metrics.json')
p.write_text(json.dumps(OUT,indent=2,allow_nan=False)+"\n")
print(json.dumps({
 'A_robust':A_parent,
 'A_child_n4':A_child['n']['4'],
 'B_3KL':B_parent,
 'B_radius_1e6':B_child['u_over_eps_grid']['1000000.0'],
 'B_output_bound':B_child['single_key_output_change_bound_unit_V_s10'],
 'C_sphere_N65536':C_parent['random_uniform_unit_sphere_query_key']['N']['65536'],
 'C_certificate':C_child['synthetic_certificate'],
 'C_memory':C_child['memory_bytes_per_KV_head']},indent=2))