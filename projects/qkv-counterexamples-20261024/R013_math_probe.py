"""R013: six distinct QKV mathematical witnesses; CPU only, no pretrained model.
Research snapshot source: Yunbo-max/new@69c37bb56cf45d4374e4acb40b81b9c3704ead07.
Actual paper comparison: MomentKV arxiv 2606.01563v1 Algorithm 1 and Eq (9),(11),(12).
"""
import json, math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, hyp0f1

SEED=20261024
rng=np.random.default_rng(SEED)

# A parent: each per-head error may shrink while an exactly cancelling readout deteriorates.
z=rng.standard_normal(100000)
e1,e2=z,-z
old_readout=e1+e2
new_e1=np.zeros_like(z); new_e2=e2
A_parent={
 'before_head1_MSE':float(np.mean(e1*e1)),
 'after_head1_MSE':float(np.mean(new_e1*new_e1)),
 'before_head2_MSE':float(np.mean(e2*e2)),
 'after_head2_MSE':float(np.mean(new_e2*new_e2)),
 'before_fused_MSE':float(np.mean(old_readout**2)),
 'after_fused_MSE':float(np.mean((new_e1+new_e2)**2)),
 'independent_nominal_95pct_joint_16':.95**16,
 'independent_nominal_95pct_joint_448':.95**448,
}
assert A_parent['after_head1_MSE']<A_parent['before_head1_MSE']
assert A_parent['before_fused_MSE']==0 and A_parent['after_fused_MSE']>.98

# A child: prior rho0=0.3 fit infinite labels, true rho drifts.
rho0=.3; g=.25; alpha=1-2*rho0
rho_grid=[.2,.3,.4,.45,.5,.6]
A_child={'nominal_rho':rho0,'nominal_amplitude':alpha,'unlabeled_excess_MSE':2*g*g,'rho_grid':{}}
for rho in rho_grid:
    risk=2*g*g*(1+alpha**2-2*alpha*(1-2*rho))
    A_child['rho_grid'][str(rho)]={'corrected_excess_MSE':risk,'worse_than_unlabeled':bool(risk>2*g*g+1e-14)}
A_child['harm_threshold_true_rho']=.5-alpha/4
assert A_child['rho_grid']['0.45']['worse_than_unlabeled']
assert not A_child['rho_grid']['0.3']['worse_than_unlabeled']
# Robust minimax over true rho in [.2,.6] chooses zero amplitude.
A_child['minimax_amplitude_for_rho_interval_.2_.6']=0.

# B parent: 2-token per-query oracle exact, one global temperature cannot match both queries.
eta=.2
k=np.eye(2)
queries=np.array([[1.,-1.],[2.,-.2]])  # q/sqrt(d) effective
scores=queries@k.T
factor=np.array([2.,.7])**eta
new_scores=scores*factor
row_exact_alpha=(new_scores[:,0]-new_scores[:,1])/(scores[:,0]-scores[:,1])
target=softmax(new_scores,axis=1)
def row_KL(alpha,row):
    model=softmax(alpha*scores[row]);p=target[row]
    return float(np.sum(p*np.log(p/model)))
opt=minimize_scalar(lambda al: sum(row_KL(al,j) for j in range(2)),bounds=(.1,3),method='bounded',options={'xatol':1e-13})
B_parent={'queries':queries.tolist(),'keys':k.tolist(),'shared_radii':[2.,.7],
 'eta':eta,'per_query_exact_temperatures':row_exact_alpha.tolist(),
 'best_global_temperature':float(opt.x),'best_global_two_query_KL_sum':float(opt.fun),
 'per_query_oracle_KL_sum':sum(row_KL(row_exact_alpha[j],j) for j in range(2)),
 'finite_shared_key_factors':factor.tolist()}
assert B_parent['best_global_two_query_KL_sum']>1e-5
assert abs(B_parent['per_query_oracle_KL_sum'])<1e-12

# B child: BF16 K radius collision yet linear V path retains exact power-of-two scaling.
import torch
torch.manual_seed(SEED)
def qwen3_norm_order(x):
    org=x.dtype; xf=x.float()
    return (xf*torch.rsqrt(xf.pow(2).mean(dim=-1,keepdim=True)+1e-6)).to(org)
x=torch.ones(128,dtype=torch.bfloat16);x2=x*2
k1,k2=qwen3_norm_order(x),qwen3_norm_order(x2)
W=torch.randn((128,128),dtype=torch.float32)/math.sqrt(128)
v1=(W@x.float()).to(torch.bfloat16);v2=(W@x2.float()).to(torch.bfloat16)
B_child={'bf16_norm_K_collides':bool(torch.equal(k1,k2)),
 'bf16_linear_V_distinct':bool(not torch.equal(v1,v2)),
 'bf16_value_norm_1':float(torch.linalg.vector_norm(v1.float())),
 'bf16_value_norm_2':float(torch.linalg.vector_norm(v2.float())),
 'ratio_linear_V_after_dtype':float(torch.linalg.vector_norm(v2.float())/torch.linalg.vector_norm(v1.float())),
 'conditional_full_KV_injectivity':'if aggregated W_V has full column rank in real arithmetic'}
assert B_child['bf16_norm_K_collides'] and B_child['bf16_linear_V_distinct']

# C parent: even full dense attention loses planted key mass with fixed logit cap as N grows.
d=128;L=math.sqrt(d);M=float(hyp0f1(d/2,d/4))
C_parent={'head_dim':d,'assumed_key_query_gain_upper':1.,'score_ceiling':L,
 'isotropic_background_E_exp_score':M,'half_attention_target_token_count':math.exp(L)/M,'N':{}}
for N in (8192,32768,65536,131072,262144,1048576):
    C_parent['N'][str(N)]={'p_target_at_perfect_alignment':math.exp(L)/(math.exp(L)+N*M),
 'log_partition_threshold':math.log(N)+math.log(M)}
assert C_parent['N']['65536']['p_target_at_perfect_alignment']<.5

# C child: MomentKV same stored moments, opposite safe correction direction.
# In d_k=2, K_R=(10 sqrt(2),0), K_evicted=(0,k sqrt(2)) with k=-1,0,+1.
# Original query qold=(1,0); test qnew=(0,1); the original scaled logits are old (10,0,0,0), test (0,-1,0,+1).
SQRT2=math.sqrt(2)
def momentkv_evict(k_all,v_all,q_old,budget=1):
    retained=list(range(len(v_all)));evicted=[]; trace=[]
    while len(retained)>budget:
        logits=np.array([float(np.dot(q_old,k_all[j])/SQRT2) for j in retained]);alpha=softmax(logits)
        if len(evicted)==0:mu=np.array([0.]);cov=np.zeros((1,2))
        else:
            ke=k_all[evicted];ve=v_all[evicted][:,None]
            mu=np.mean(ve,axis=0)
            cov=sum((ve[i]-mu)[:,None]*(ke[i]-ke.mean(axis=0))[None,:] for i in range(len(evicted)))
        residual=[]
        for j in retained:
            if len(evicted)==0:r=v_all[j]
            else:r=float(v_all[j]-mu[0]-(cov@k_all[j])[0]/(len(evicted)*SQRT2))
            residual.append(abs(r))
        scores=alpha*np.array(residual)
        idx=int(np.argmin(scores));picked=retained.pop(idx);evicted.append(picked)
        trace.append({'evicted_token_index':picked,'retained_candidates':list(retained),'selected_score':float(scores[idx])})
    return retained,evicted,trace

k_scalar=np.array([-1.,0.,1.]);A=8.;key_matrix=np.vstack([np.array([0.,k*SQRT2]) for k in k_scalar])
all_keys=np.vstack([np.array([10*SQRT2,0.]),key_matrix]);q_old=np.array([1.,0.]);qnew=np.array([0.,1.]);
worlds={}
for sign in (+1,-1):
    values=k_scalar+sign*A*np.array([1.,-2.,1.])
    vals=np.r_[2.,values]
    retain,evict,trace=momentkv_evict(all_keys,vals,q_old,1)
    evk=all_keys[evict];evv=vals[evict]
    stats={'n':len(evict),'sum_keys':evk.sum(axis=0).tolist(), 'sum_values':float(evv.sum()),
           'sum_value_outer_keys':(evv[:,None]*evk).sum(axis=0).tolist()}
    logits=all_keys@qnew/SQRT2
    actual=float(np.dot(softmax(logits),vals))
    # MomentKV Eq 9/11/12: E count3, mean key=0, mean value0, sum vk slope2.
    estimated=1.
    baseline=2.
    worlds[str(sign)]={'retained_indices':retain,'evicted_indices':evict,'eviction_trace':trace,
       'moments':stats,'full_attention_output':actual,'plain_eviction_output':baseline,'moment_output':estimated,
       'plain_abs_error':abs(baseline-actual),'moment_abs_error':abs(estimated-actual),
       'derivative_of_squared_error_at_lambda_0':2*(estimated-baseline)*(baseline-actual)}
assert worlds['1']['retained_indices']==worlds['-1']['retained_indices']==[0]
assert worlds['1']['moments']==worlds['-1']['moments']
assert worlds['1']['full_attention_output']>2 and worlds['-1']['full_attention_output']<2
# r=2, d= h-r = -1. For sign +, y>r => any move r-lambda hurts.
for lam in (.0001,.1,.5,1.):
    new=2-lam
    assert abs(new-worlds['1']['full_attention_output'])>abs(2-worlds['1']['full_attention_output'])
C_child={'worlds':worlds,'common_plain_output':2.,'common_moment_output':1.,
 'result':'no strictly positive moment-direction update can be uniformly safe across these indistinguishable states',
 'minimal_robust_nonnegative_lambda':0.}

result={'execution':{'seed':SEED,'numpy':np.__version__,'scipy':__import__('scipy').__version__,'torch':torch.__version__,'device':'CPU'},
 'A_parent':A_parent,'A_child':A_child,'B_parent':B_parent,'B_child':B_child,'C_parent':C_parent,'C_child':C_child}
print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))