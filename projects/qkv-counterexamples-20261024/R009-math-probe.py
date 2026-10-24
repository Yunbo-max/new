"""R009 six-leaf mathematics diagnostic: CPU only, no LLM inference or GPU."""
import json, math
import numpy as np
import torch
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

# A parent: context-conditioned variance minimization is unsafe although global covariance is zero.
# X in {-1,+1}, theta=X*z, y1=theta+eps, D=z+u, z,u,eps ~ N(0,1).
A_parent = {"unconditional_cov_theta_D":0.,"task_opt_beta_each_context":0.,
            "variance_opt_beta_X_plus":-0.5,"variance_opt_beta_X_minus":0.5,
            "true_test_MSE_task_opt":1.,"true_test_MSE_variance_opt":1.5,
            "observed_conditional_variance_baseline":2.,"observed_conditional_variance_proxy_opt":1.5}
assert A_parent['true_test_MSE_variance_opt']>A_parent['true_test_MSE_task_opt']
# A child: exact sign-identification probability from labels and Gaussian posterior.
T=9.; sigma_label_sq=1.;g=.25
post_theta_var=T*.5/(T+.5)
sigma_sq=post_theta_var+sigma_label_sq
A_child={"task_sign_magnitude":g,"label_effective_noise_variance":sigma_sq,"n":{}}
for n in (4,16,64,256):
    threshold=g*math.sqrt(2*n/sigma_sq)
    pe=float(student_t.cdf(-threshold,df=n))
    regret=8*g*g*pe
    A_child["n"][str(n)]={"sign_error_prob":pe,"sign_classifier_excess_MSE":regret,"unlabeled_minimax_excess_MSE":2*g*g,"labels_beat_unlabeled":pe<.25}
assert A_child['n']['256']['sign_error_prob'] < A_child['n']['4']['sign_error_prob']
# B parent: second-order Fisher-KL residual after optimizing a row temperature.
s=np.array([1.,0.,-1.]); radii=np.array([2.,1.,.7]); a=np.log(radii); u=s*a
p=softmax(s)
E=lambda v:float(p@v)
var=lambda v: E(v*v)-E(v)**2
cov=lambda v,w:E(v*w)-E(v)*E(w)
projection=cov(u,s)/var(s)
fisher_residual=var(u)-cov(u,s)**2/var(s)
assert fisher_residual>0
B_parent={"s":s.tolist(),"radius_ratios":radii.tolist(),"temp_tangent":projection,
          "fisher_residual_coefficient":fisher_residual,"eta_probe":{}}
for eta in (.025,.05,.10,.20):
    changed=softmax(s*np.exp(eta*a))
    def kl(alpha):
        q=softmax(alpha*s)
        return float(np.sum(changed*np.log(changed/q)))
    opt=minimize_scalar(kl,bounds=(.01,3),method='bounded',options={'xatol':1e-13})
    B_parent['eta_probe'][str(eta)]={"minimum_KL":max(0,float(opt.fun)),"optimal_temperature":float(opt.x),
           "quadratic_KL":.5*eta*eta*fisher_residual,
           "constant_value_output_difference":0.}
assert B_parent['eta_probe']['0.025']['minimum_KL']>0
# B child: Qwen3 source-order RMSNorm; exactly the change at dtype cast, unit learned gain.
def qnorm(x,eps=1e-6):
    dtype=x.dtype; x=x.float();return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)).to(dtype)
torch.manual_seed(20261024)
B_child={"head_dim":128,"epsilon":1e-6,"unit_direction_values":{},"random_1000x128":{}}
for dtype in (torch.float16,torch.bfloat16,torch.float32):
    x=torch.ones((128,),dtype=dtype)
    y1=qnorm(x);y2=qnorm(2*x)
    nmatch=bool(torch.equal(y1,y2))
    B_child['unit_direction_values'][str(dtype)]={"same_after_cast":nmatch,
         "first_x1":float(y1[0]),"first_x2":float(y2[0])}
    rand=torch.randn((1000,128),dtype=torch.float32).to(dtype)
    is_same=torch.all(qnorm(rand)==qnorm(2*rand),dim=-1)
    B_child['random_1000x128'][str(dtype)]={"entire_vector_match_rate":float(is_same.float().mean()),
        "total_component_match_rate":float((qnorm(rand)==qnorm(2*rand)).float().mean())}
assert B_child['unit_direction_values']['torch.float16']['same_after_cast']
assert B_child['unit_direction_values']['torch.bfloat16']['same_after_cast']
assert not B_child['unit_direction_values']['torch.float32']['same_after_cast']
# C parent: vanishing mean logit variance with persistent softmax contribution.
C_parent={"N_grid":{},"limit":{str(c):1/(1+math.exp(-c)) for c in (-2,0,2)}}
for N in (32,2048,65536):
    C_parent['N_grid'][str(N)]={}
    for c in (-2,0,2):
        q=math.log(N)+c
        f=2*math.sinh(q)/(N+2*math.cosh(q))
        variance=2*q*q/(N+2)
        C_parent['N_grid'][str(N)][str(c)]={"output":f,"mean_squared_logits":variance}
assert abs(C_parent['N_grid']['65536']['0']['output']-.5)<1e-6
# C child: deterministic sufficient certificate for any full output consistent with moment stats
# Assumptions: |v_i|<=Vmax, |q(k_i-kbar)/sqrt(d)|<=eta, 2d value/key scalar example.
def certificate(t,b):
    # evicted keys [-t,+t], evicted values [-1,+1], retained logit zero and value b.
    eta=abs(t)
    vmax=max(1.,abs(b))
    zhat=2.;zr=1.;what=zhat/(zr+zhat)
    # Upper bound Z_E/ Zhat from centered deviations.
    rmax=min(math.exp(eta),1+.5*math.exp(eta)*eta*eta)
    wmax=zhat*rmax/(zr+zhat*rmax)
    dw=wmax-what
    # Bound ||f_E - fhat_E|| <= exp(eta)/2 *m2*(B+|Cq|), m2<=eta²,
    # B=max_i |v_i-vbar| <=2*Vmax.
    e_bound=.5*math.exp(eta)*eta*eta*(2*vmax+abs(t))
    error_bound=2*vmax*dw+what*e_bound
    full=(b+2*math.sinh(t))/(1+2*math.cosh(t))
    corrected=(b+2*t)/3
    correction=corrected-b
    return {"t":t,"retained_value":b,"estimated_output":corrected,"true_output":full,
            "plain_abs_error":abs(b-full),"corrected_abs_error":abs(corrected-full),
            "predicted_move_abs":abs(correction),"computable_error_bound":error_bound,
            "sufficient_safety_certificate":error_bound<abs(correction)/2,
            "correction_really_helps":abs(corrected-full)<abs(b-full)}
C_child={"safe_small_shift":certificate(.05,1.),
         "unsafe_large_shift":certificate(8.,4.),
         "unsafe_zero_plain_loss":certificate(1.,math.tanh(1))}
assert C_child['safe_small_shift']['sufficient_safety_certificate']
assert C_child['safe_small_shift']['correction_really_helps']
assert not C_child['unsafe_large_shift']['sufficient_safety_certificate']
assert not C_child['unsafe_large_shift']['correction_really_helps']
# six independent leaf checks, no Work subagents or native model.
out={'A_parent':A_parent,'A_child':A_child,'B_parent':B_parent,
     'B_child':B_child,'C_parent':C_parent,'C_child':C_child,
     'environment':{'numpy':np.__version__,'torch':torch.__version__}}
print(json.dumps(out,indent=2,allow_nan=False))
