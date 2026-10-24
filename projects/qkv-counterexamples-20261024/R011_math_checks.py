"""R011 six-node falsification witnesses. CPU-only: no pretrained inference, no GPU.
Fully deterministic seed; use independent simple analytical controls.
"""
import json, math
import numpy as np
import torch
from scipy.special import softmax
from scipy.optimize import brentq

rng=np.random.default_rng(20261024)
out={}

# A(parent): a minimax coefficient is safe only under the declared target-covariance interval.
a=.75;C=.4;v=2.
beta_rob=-(a-C)/v
risk_delta=lambda beta,c:v*beta**2+2*beta*(a-c)
out['A_parent']={
 'observed_cov_a':a,'uncertainty_bound_C':C,'robust_beta':beta_rob,
 'worst_case_inside_bound':risk_delta(beta_rob,C),
 'out_of_bound_target_cov':.8,
 'shifted_excess_true_MSE':risk_delta(beta_rob,.8),
 'no_change_excess_MSE':0.
}
assert risk_delta(beta_rob,C)<0<risk_delta(beta_rob,.8)

# A(child): Bayes sign inference assumes the observed task labels obey its likelihood.
# Under dataset-wide sign-inverted labels, infinite data drives the posterior to -s.
T=9.;g=.25;sigmasq=1+T*.5/(T+.5)
R=200000
rho_values=(0,.1,.3,.5)
N_values=(4,16,64,256)
label_results={}
for n in N_values:
 S=2*rng.chisquare(n,size=R)
 Z=rng.standard_normal(R)
 per_scenario=[]
 for flipped in [False, True]:
  # the TRUE sign is +1 by symmetry; if flipped, labels look like sign -1
  sum_dr=((-1 if flipped else +1)*g*S+math.sqrt(sigmasq)*np.sqrt(S)*Z)
  mu=np.tanh((g/sigmasq)*sum_dr)
  regret=2*g*g*np.mean((mu-1)**2)
  per_scenario.append(float(regret))
 label_results[str(n)]={'correct_likelihood_regret':per_scenario[0],
                        'inverted_likelihood_regret':per_scenario[1],
                        'by_domain_inversion_fraction':{str(rho):float((1-rho)*per_scenario[0]+rho*per_scenario[1]) for rho in rho_values}}
label_results['no_label_midpoint_excess']=2*g*g
label_results['infinite_n_misspecified_limit']={str(rho):8*g*g*rho for rho in rho_values}
assert label_results['256']['inverted_likelihood_regret']>3*label_results['no_label_midpoint_excess']
out['A_child']=label_results

# B(parent): GQA two query heads, identical head maps but equal/opposite output projection blocks.
s=np.array([1.,0.,-1.]);radii=np.array([2.,1.,.7]);eta=.3
V=np.array([1.,0.,-1.])
p0=softmax(s);p1=softmax(s*radii**eta)
y0=float(p0@V);y1=float(p1@V)
kl=float(np.sum(p1*np.log(p1/p0)))
# Two DISTINCT query heads sharing K,V and using different query magnitudes.
# By continuity find q2>q1 for which nonzero radial output changes coincide.
def head(t,changed=False):
 return float(softmax(t*s*(radii**eta if changed else np.ones(3)))@V)
q1=1.
change=lambda t: head(t,True)-head(t,False)
q2=brentq(lambda t:change(t)-change(q1),1.3,5.)
W=np.array([1.,-1.])
group0=head(q1)-head(q2)
group1=head(q1,True)-head(q2,True)
out['B_parent']={'per_head_KL':kl,'head_1_query_scale':q1,'head_2_query_scale':q2,
  'head_1_baseline_output':head(q1),'head_2_baseline_output':head(q2),
  'head_1_perturbation_delta':change(q1),'head_2_perturbation_delta':change(q2),
  'GQA_group_baseline_after_WO':group0,'GQA_group_perturbed_after_WO':group1,
  'GQA_group_post_WO_delta':group1-group0,'output_blocks':[1,-1],
  'status':'Distinct non-identical Q heads, same K/V, no pretrained weights; no model claim'}
assert kl>0 and abs(q2-q1)>.5 and abs(group0)>.1 and abs(change(q1))>.04 and abs(group1-group0)<1e-12

# B(child): phase transition in finite-precision cast collisions for input radius vs epsilon.
eps=1e-6;torch.manual_seed(20261024)
x0=torch.randn(1000,128,dtype=torch.float32)
x0=x0/torch.sqrt(x0.square().mean(-1,keepdim=True))
ratios=[1,10,100,1000,10000,100000,1000000,100000000]
b_table=[]
def norm(x):
 y=x.float();return (y*torch.rsqrt(y.square().mean(-1,keepdim=True)+eps)).to(x.dtype)
for ratio in ratios:
 d={'input_mean_square_over_eps':ratio,'radial_inverse_condition':ratio+1,
    'ideal_normalized_x_to_2x_rel_change':2*math.sqrt(ratio+1)/math.sqrt(4*ratio+1)-1}
 for dtype in [torch.bfloat16,torch.float16,torch.float32]:
  x=(x0*math.sqrt(eps*ratio)).to(dtype)
  frac=float(torch.all(norm(x)==norm(2*x),dim=-1).float().mean())
  d[str(dtype).replace('torch.','')+'_full_vector_collision_rate']=frac
 b_table.append(d)
out['B_child']={'head_dim':128,'eps':eps,'samples_per_condition':1000,'scale_sweep':b_table}
assert b_table[0]['bfloat16_full_vector_collision_rate']==0
assert b_table[-1]['bfloat16_full_vector_collision_rate']>0.95

# C(parent): iid sphere background cannot form near-max logits, but a planted aligned key can.
# unit gain, q and keys norms=sqrt(d), aligned score sqrt(d); unrelated scores approximated N(0,1).
d=128;N=65536;aligned_score=math.sqrt(d)
aligned_exp=math.exp(aligned_score)
# Exact distribution of one coordinate of a random uniform sphere direction at d128.
_z=rng.standard_normal((120,N))
_chi2=rng.chisquare(d-1,size=(120,N))
background_logit=math.sqrt(d)*_z/np.sqrt(_z*_z+_chi2)
Zbg=np.exp(background_logit).sum(axis=1)
planted_p=aligned_exp/(Zbg+aligned_exp)
null_required_cos=math.log(N)/math.sqrt(d)
out['C_parent']={'d':d,'background_tokens':N,'planted_score':aligned_score,
 'necessary_cosine_for_lnN':null_required_cos,
 'one_planted_key_softmax_mass_mean':float(planted_p.mean()),
 'one_planted_key_softmax_mass_p05':float(np.quantile(planted_p,.05)),
 'one_planted_key_softmax_mass_p95':float(np.quantile(planted_p,.95)),
 'background_distribution':'exact iid isotropic uniform-sphere key marginal; planted key exactly aligned with q',
 'analytical_mass_with_E_expZapprox_exp_half':aligned_exp/(N*math.exp(.5)+aligned_exp)}
assert .3<out['C_parent']['one_planted_key_softmax_mass_mean']<.6

# C(child): same stored MomentKV moments imply opposite preferences for an output correction.
# Exact k=(-1,0,1), value alternatives k +/- 4*(1,-2,1), one retained value 2 at logit0.
k=np.array([-1.,0.,1.]);b=2.;A=4.
scenarios={}
for sign in (-1,+1):
 value=k+sign*A*np.array([1.,-2.,1.]);Z=np.exp(np.r_[0.,k]);
 y=float((Z*np.r_[b,value]).sum()/Z.sum())
 simple=abs(y-b); corrected=abs(y-1.)
 scenarios[str(sign)]={'stored_moments':[len(k),float(k.sum()),float(value.sum()),float((k*value).sum())],
    'full_attention_output':y,'plain_eviction_abs_error':simple,'moment_correction_abs_error':corrected,
    'correction_helps':corrected<simple}
assert scenarios['-1']['stored_moments']==scenarios['1']['stored_moments']
assert scenarios['-1']['correction_helps']!=scenarios['1']['correction_helps']
# Optimal choice under prior p=P(sign=-1), with L1 error loss.
plus=scenarios['1'];minus=scenarios['-1']
plus_harm=plus['moment_correction_abs_error']-plus['plain_eviction_abs_error']
minus_gain=minus['plain_eviction_abs_error']-minus['moment_correction_abs_error']
threshold=plus_harm/(plus_harm+minus_gain)
out['C_child']={'worlds':scenarios,'harm_from_correction_in_plus':plus_harm,
 'benefit_from_correction_in_minus':minus_gain,
 'bayes_correction_threshold_prior_P_minus':threshold,
 'required_extra_information':'at least a bit resolving the hidden world for a pointwise optimal gate; statistic alone cannot do so',
 'new_oct_2026_prior_art_access':'Perturbed Retrieval certificate retains/scans all keys, distinct from permanent moment-based eviction'}
assert 0<threshold<1
print(json.dumps(out,indent=2))