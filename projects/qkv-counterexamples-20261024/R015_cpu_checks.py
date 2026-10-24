"""R015 full-frontier CPU-only mathematical checks.

This is a synthetic, scoped diagnostic. NO pretrained model, GPU, private API,
formal reviewer, or author MomentKV implementation is executed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom
from scipy.special import expit
import torch

SEED = 20261024
rng = np.random.default_rng(SEED)


def scalar(x):
    return float(x)


# A: an optimal *partial* correction of one head after combining both errors.
# e1 ~ N(0,1), e2=-c e1 + eta, eta~N(0,0.6^2).
# R(lam) = (1-lam-c)^2 + 0.36, lam in [0,1].
A_parent = {}
for c in [0.8, 1.2]:
    lam_star = float(np.clip(1-c, 0, 1))
    analytic = lambda lam: (1-lam-c)**2 + .36
    e1 = rng.normal(size=100000)
    eta = rng.normal(scale=.6, size=100000)
    e2 = -c*e1+eta
    empirical = {
        'no_change':float(np.mean((e1+e2)**2)),
        'partial_oracle':float(np.mean(((1-lam_star)*e1+e2)**2)),
        'full_repair':float(np.mean(e2**2)),
    }
    exact = dict(no_change=analytic(0),partial_oracle=analytic(lam_star),full_repair=analytic(1))
    assert abs(empirical['no_change']-exact['no_change'])<.013
    assert abs(empirical['partial_oracle']-exact['partial_oracle'])<.013
    assert abs(empirical['full_repair']-exact['full_repair'])<.03
    if c==.8:
        assert exact['partial_oracle'] < exact['no_change'] < exact['full_repair']
    else:
        assert exact['no_change']==exact['partial_oracle'] and exact['full_repair']>exact['no_change']
    A_parent[str(c)] = {'cov_e1e2':-c, 'optimal_partial_lambda':lam_star,'exact_mse':exact,'cpu_mse':empirical}

# A child: m *trusted, independent domain flip anchors* estimate rho;
# Hoeffding UCB with P(true rho > UCB) <= delta, and shrink kappa=(1-2 UCB)_+.
# g=.25, rho=.30; test loss against hidden original sign.
A_child = {}
g = .25
rho = .3
delta = .05
baseline=2*g*g
for m in [16,32,64,128,256]:
    k=np.arange(m+1)
    pmf=binom.pmf(k,m,rho)
    margin=math.sqrt(math.log(1/delta)/(2*m))
    upper=np.minimum(1,k/m+margin)
    shrink=np.maximum(0,1-2*upper)
    risk=2*g*g*(1+shrink*shrink-2*shrink*(1-2*rho))
    coverage=scalar(np.sum(pmf[upper>=rho]))
    action=scalar(np.sum(pmf[shrink>0]))
    avg=scalar(np.sum(pmf*risk))
    assert coverage>=.95-1e-12
    A_child[str(m)]={'margin':margin,'expected_risk':avg,'no_label_risk':baseline,
                     'prob_nonzero_action':action,'coverage_probability':coverage,
                     'prob_risk_worse_than_no_label':scalar(np.sum(pmf[risk>baseline+1e-12]))}
A_child['nominal_n_to_enable_positive']=math.log(1/delta)/(2*(.5-rho)**2)
A_child['nominal_n_for_shrink_ge_0_1']=math.log(1/delta)/(2*(.45-rho)**2)

# B: shared global temperature vs privileged per-query temperature in a fixed 2D output.
# g_i=(1,0), radial tangent h_i=c_i*g_i+(0,0.3), eta=0.1.
cs=np.array([-1.,0.,1.,2.]); h=np.stack([cs,np.full_like(cs,.3)],axis=1)
gt=np.tile(np.array([1.,0.]),(4,1)); eps=.1
row_coeff=np.einsum('ij,ij->i',h,gt)/np.sum(gt**2,axis=1)
global_coeff=float(np.sum(h*gt)/np.sum(gt*gt))
row_residual=eps*eps*float(np.sum((h-row_coeff[:,None]*gt)**2))
global_residual=eps*eps*float(np.sum((h-global_coeff*gt)**2))
between_query_gap=eps*eps*float(np.sum(np.sum(gt*gt,axis=1)*(row_coeff-global_coeff)**2))
assert abs(global_residual-row_residual-between_query_gap)<1e-12
assert abs(row_residual-.0036)<1e-12 and abs(global_residual-.0536)<1e-12
B_parent={'per_query_temperature_tangents':row_coeff.tolist(), 'global_temperature_tangent':global_coeff,
    'per_query_oracle_squared_error':row_residual,'global_fitted_squared_error':global_residual,
    'extra_shared_parameter_penalty':between_query_gap,'positive_temperature_ranges_at_eta_0_1':(1+eps*row_coeff).tolist()}

# B child: actual Qwen-style dtype cast can erase head-local radial KV,
# but a separate cross-head Value projection can preserve it. This is a toy linear algebra interface.
# Wk selects coords 3,4, Wv(local) selects coords 1,2 of a 6-D hidden input.
Wk=np.zeros((2,6)); Wk[0,2]=1;Wk[1,3]=1
Wv=np.zeros((2,6)); Wv[0,0]=1;Wv[1,1]=1
h0=np.array([0.,0.,1.,1.,0.,0.]); h2=2*h0

def head_qknorm(key):
    x=torch.tensor(key,dtype=torch.bfloat16)
    x32=x.to(torch.float32)
    out=(x32*torch.rsqrt(torch.mean(x32**2)+1e-6)).to(torch.bfloat16)
    return out

k0=head_qknorm(Wk@h0);k2=head_qknorm(Wk@h2)
assert torch.equal(k0,k2)
assert np.allclose(Wv@h0,Wv@h2)
Wv_second=np.zeros((2,6));Wv_second[0,2]=1; Wv_second[1,3]=1
assert np.linalg.norm(Wv_second@h0-Wv_second@h2)>0
B_child={'hidden_dim_toy':6,'local_value_rank':int(np.linalg.matrix_rank(Wv)),
         'local_value_nullity':int(6-np.linalg.matrix_rank(Wv)),
         'qk_cast_keys_equal':bool(torch.equal(k0,k2)),
         'local_values_equal':bool(np.allclose(Wv@h0,Wv@h2)),
         'other_value_head_norm_difference':float(np.linalg.norm(Wv_second@h0-Wv_second@h2)),
         'Qwen3_0p6B_possible_one_value_head_nullity_at_least':1024-128,
         'Qwen3_0p6B_all_8_value_heads_potentially_full_rank':True}

# C parent, a 2-class readout: an important target with p=.432 may not change decision
# without projected Value *logit-margin contrast* a. Also show paged-cache occupancy.
pstar=.432
base_margin=-.2
C_parent={'base_decision_margin':base_margin,'attention_mass':pstar,'value_logit_contrasts':{}}
for d in [-1,0,.25,1,2]:
    target_margin=base_margin+pstar*d
    C_parent['value_logit_contrasts'][str(d)]={'final_margin':target_margin,
        'positive_class_probability':scalar(expit(target_margin)),
        'flips_winner_relative_to_base':bool(target_margin>0),
        'delta_margin':pstar*d}
assert not C_parent['value_logit_contrasts']['0.25']['flips_winner_relative_to_base']
assert C_parent['value_logit_contrasts']['1']['flips_winner_relative_to_base']
N_blocks=1024;B=16; N_tokens=N_blocks*B; survivors=1024
# 1024 survivors (1/16 logical occupancy); as one survivor in each page, physically all pages remain.
random_pages=[]
for _ in range(1000):
    kept=rng.choice(N_tokens,size=survivors,replace=False)
    random_pages.append(len(np.unique(kept//B)))
C_parent['paged_memory']={
    'blocks_before':N_blocks,'tokens_before':N_tokens,'logical_surviving_tokens':survivors,
    'logical_fraction':survivors/N_tokens,'worst_spread_used_blocks':N_blocks,
    'best_compacted_used_blocks':survivors//B,
    'random_selection_mean_used_blocks':float(np.mean(random_pages)),
    'random_selection_sd_used_blocks':float(np.std(random_pages)),
    'random_selection_mean_physical_fraction':float(np.mean(random_pages)/N_blocks)}
assert C_parent['paged_memory']['worst_spread_used_blocks']==N_blocks
assert 590<C_parent['paged_memory']['random_selection_mean_used_blocks']<700

# C child: identical n, sum K, sum V, sum K*V summaries with unbounded true
# exponential output when Value magnitude has no separately retained bound.
t=1.;keys=np.array([-1.,0.,1.]); base=np.array([1.,-2.,1.]);
C_child={'test_score_t':t,'kept_token':{'key':0,'value':0},'worlds':{}}
for A in [1,10,100]:
    world={}
    for sign in [-1,1]:
        values=sign*A*base
        saved=[float(len(keys)),float(np.sum(keys)),float(np.sum(values)),float(np.dot(keys,values))]
        full=float(np.sum(np.exp(t*keys)*values)/(1+np.sum(np.exp(t*keys))))
        world[str(sign)]={'retained_stats':saved,'full_attention_output':full,'summary_only_prediction':0.,
                          'prediction_abs_error':abs(full)}
    assert world['1']['retained_stats']==world['-1']['retained_stats']
    assert abs(world['1']['full_attention_output']+world['-1']['full_attention_output'])<1e-12
    C_child['worlds'][str(A)]=world
slope=(math.cosh(t)-1)/(1+math.cosh(t))
assert abs(C_child['worlds']['100']['1']['full_attention_output']-100*slope)<1e-11
C_child['unbounded_output_slope_per_unit_amplitude']=slope
C_child['full_output_when_max_value_norm_is_Vmax_bound']=lambda_hint = (slope/2)
# above number times Vmax, because values with max abs value 2A impose A<=Vmax/2.
C_child['qualification']='Applies only when the summarized-interface has no certified global or stored Value norm bound; model weights/normalization can supply a finite but loose global bound.'

results={'schema':'r015-cpu-math-v1','source_project_revision':14,'seed':SEED,
    'execution':'local CPU synthetic analytical checks only, not Qwen inference',
    'A_parent':A_parent,'A_child':A_child,'B_parent':B_parent,'B_child':B_child,
    'C_parent':C_parent,'C_child':C_child,
    'python':sys.version.split()[0],'numpy':np.__version__,'torch':torch.__version__}
if __name__=='__main__':
    output=Path(__file__).with_name('R015_metrics.json')
    output.write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','six_nodes':list(k for k in results if k in ['A_parent','A_child','B_parent','B_child','C_parent','C_child']),
      'A_partial':A_parent,'A_anchor_action':{x:v['prob_nonzero_action'] for x,v in A_child.items() if isinstance(v,dict)},
      'B_shared_penalty':B_parent['extra_shared_parameter_penalty'],
      'C_paged_mean_occupied':C_parent['paged_memory']['random_selection_mean_used_blocks'],
      'C_unbounded_slope':slope,'metrics_file':str(output)},indent=2))