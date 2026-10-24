"""R007 minimal reproducible counterexample checks. Stdlib Python only.
Run: python r007_gate.py
These artificial results are NOT language-model experimental evidence.
"""
import random,math,json

def compare(p0,p1,n,c,seed,delta=.05):
 r=random.Random(seed)
 q0=sum(r.random()<p0 for _ in range(n))/n
 q1=sum(r.random()<p1 for _ in range(n))/n
 e=math.sqrt(math.log(4/delta)/(2*n))
 return (q1-q0-c,q1-q0-c-2*e)

def main():
 assert 1-.4>.5 and .3-.4<0  # B1: necessity differs from gain
 assert (.9-.1)-(.5-.5)==.8  # B1: forcing can alter continuation
 null=[compare(.5,.5,128,0,s) for s in range(100)]
 good=[compare(.15,.85,128,.05,s) for s in range(100)]
 # B2: test confidence certificate, not a new learning method.
 assert sum(y>0 for _,y in null)<=5
 assert sum(y>0 for _,y in good)>=95
 # B3: outer optimization induces visible/hidden gap without gaming.
 r=random.Random(303); v=h=fv=fh=0; n=5000
 for _ in range(n):
  vs=[r.random()<.5 for i in range(10)]
  hs=[r.random()<.5 for i in range(10)]
  k=max(range(10),key=lambda i:vs[i])
  v+=vs[k];h+=hs[k];fv+=vs[0];fh+=hs[0]
 assert (v-h)/n>.45 and abs(fv-fh)/n<.05
 # B4: assigned budget helps while consumed-token correlation reverses.
 budget1=(.9+.2)/2; budget3=(.95+.6)/2
 used1=(.9+.2+.95)/3;used3=.6
 assert budget3>budget1 and used3<used1
 print(json.dumps({"synthetic_only":True,"B1_forcing_bias":.8,
  "B2_null_point_calls":sum(x>0 for x,_ in null),
  "B2_null_certified_calls":sum(y>0 for _,y in null),
  "B2_beneficial_certified_calls":sum(y>0 for _,y in good),
  "B3_winner_gap":(v-h)/n,"B3_fixed_gap":(fv-fh)/n,
  "B4_assigned_budget_gain":budget3-budget1,
  "B4_consumed_tokens_association":used3-used1},indent=2))
if __name__=="__main__":main()
