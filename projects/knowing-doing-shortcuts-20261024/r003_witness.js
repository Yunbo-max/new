// Autopilot Manual Wake R003 – standalone artificial witness suite.
// Run: node r003_witness.js (Node.js >= 18); no external dependencies.
// Demonstrates algebraic counterexamples ONLY; not an LLM experiment.

function B1() {
 const M=[[1,100],[0,1]];
 const probe0=[1,0], act0=[0,1];
 const minvT=[[1,0],[-100,1]];
 const mult=(A,v)=>A.map(r=>r[0]*v[0]+r[1]*v[1]);
 const dot=(a,b)=>a[0]*b[0]+a[1]*b[1];
 const cos=(a,b)=>dot(a,b)/Math.sqrt(dot(a,a)*dot(b,b));
 const pk=mult(minvT,probe0), pa=mult(minvT,act0);
 let maxDelta=0;
 for (const n of [0,1]) for(const z of [0,1]) {
  const h=[n,z], hh=mult(M,h);
  maxDelta=Math.max(maxDelta,Math.abs(dot(probe0,h)-dot(pk,hh)),Math.abs(dot(act0,h)-dot(pa,hh)));
 }
 const result={original_probe_action_cosine:cos(probe0,act0),reparameterized_cosine:cos(pk,pa),max_probe_action_logit_change:maxDelta,linear_basis_change_determinant:1};
 if(!(result.original_probe_action_cosine===0 && result.reparameterized_cosine < -0.9999 && maxDelta<1e-12))throw Error("B1 fail");
 return result;
}

function B2(){
 const sigmoid=x=>1/(1+Math.exp(-x));
 const worldA={v:.12,lambda:1,beta:3},worldB={v:.36,lambda:3,beta:1};
 const costs=Array.from({length:101},(_,i)=>i/500);
 const maxDifference=Math.max(...costs.map(c=>Math.abs(sigmoid(worldA.beta*(worldA.v-worldA.lambda*c))-sigmoid(worldB.beta*(worldB.v-worldB.lambda*c)))));
 const pForced1=.80,pForced0=.60;
 const deterministicCosts=[.03,.08,.13,.18];
 const actualRegrets=deterministicCosts.map(c=>(worldA.v-worldA.lambda*c<=0)?Math.max(0,pForced1-pForced0-c):0);
 const result={max_choice_probability_difference_under_rescaling:maxDifference,actual_forced_uplift:pForced1-pForced0,
 mean_regret_over_four_prices:actualRegrets.reduce((a,b)=>a+b,0)/4,
 observed_choice_identifies_subjective_value_only_up_to_noise_scale:true};
 if(maxDifference>1e-12||Math.abs(result.mean_regret_over_four_prices-.0225)>1e-12) throw Error("B2 fail");
 return result;
}

function B3(){
 const shortcutProb=(expose,exploitable)=>.10+.05*expose+.04*exploitable+.30*expose*exploitable;
 const v00=shortcutProb(0,0),v10=shortcutProb(1,0),v01=shortcutProb(0,1),v11=shortcutProb(1,1);
 const did=(v11-v01)-(v10-v00);
 const hiddenAccuracy=p=>.85-.40*p;
 const proxyScore=p=>.60+.30*p;
 const result={shortcuts:{placebo_hidden:v00,placebo_visible:v10,exploitable_hidden:v01,exploitable_visible:v11},
 difference_in_differences:did,true_success_gap_on_exploitable_visible_minus_hidden:hiddenAccuracy(v11)-hiddenAccuracy(v01),
 proxy_score_gap_on_exploitable_visible_minus_hidden:proxyScore(v11)-proxyScore(v01)};
 if(Math.abs(did-.30)>1e-12||result.true_success_gap_on_exploitable_visible_minus_hidden>=0||result.proxy_score_gap_on_exploitable_visible_minus_hidden<=0)throw Error("B3 fail");
 return result;
}

function B4(){
 const easy=[.8,.82,.84,.86],hard=[.15,.15,.15,.8],budget=3;
 const greedy=()=>{let x=0,y=0;for(let k=0;k<budget;k++){const d0=x+1<easy.length?easy[x+1]-easy[x]:-Infinity,d1=y+1<hard.length?hard[y+1]-hard[y]:-Infinity;if(d0>=d1)x++;else y++;}return {allocation:[x,y],score:easy[x]+hard[y]};};
 let optimal=-Infinity,alloc=[];
 for(let i=0;i<=budget;i++){const s=easy[i]+hard[budget-i];if(s>optimal){optimal=s;alloc=[i,budget-i];}}
 const curveIncrements=[];
 const rec=(arr)=>{if(arr.length===4){curveIncrements.push(arr.slice());return;}for(const v of [0,.02,.04,.06])if(!arr.length||v<=arr[arr.length-1])rec([...arr,v]);};
 rec([]);
 const curves=curveIncrements.map(xs=>[.10,.10+xs[0],.10+xs[0]+xs[1],.10+xs[0]+xs[1]+xs[2],.10+xs.reduce((a,b)=>a+b,0)]);
 let cases=0,violations=0;
 for(const a of curves) for(const b of curves){
   const maxBudget=4;
   let gx=0,gy=0;for(let j=0;j<maxBudget;j++){const d0=a[gx+1]-a[gx],d1=b[gy+1]-b[gy];if(gx<4&&(gy===4||d0>=d1))gx++;else gy++;}
   const greedyVal=a[gx]+b[gy];
   let best=-Infinity;for(let k=0;k<=4;k++)best=Math.max(best,a[k]+b[4-k]);
   cases++;if(best-greedyVal>1e-10)violations++;
 }
 const result={nonconcave_greedy:greedy(),nonconcave_optimal:{allocation:alloc,score:optimal},nonconcave_gap:optimal-greedy().score,
 concave_synthetic_pair_cases:cases,concave_greedy_violations:violations};
 if(violations!==0||Math.abs(result.nonconcave_gap-.59)>1e-12)throw Error("B4 fail");
 return result;
}

const output={kind:"explicit_artificial_numeric_or_logical_tests_not_LLM",B1:B1(),B2:B2(),B3:B3(),B4:B4(),execution:"Node.js script, assertions must pass"};
console.log(JSON.stringify(output,null,2));
