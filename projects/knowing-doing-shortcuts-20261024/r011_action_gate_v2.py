"""R011 gate v2 (compact GitHub companion). Full tested version in R011 ZIP.
Admits structured logs, not actual model states; validate with a real instrumented executor.
"""
import hashlib,json,math
from collections import defaultdict

FIELDS={'task','state','arm','seed','model_sha','runner_sha','tool_sha',
        'grader_sha','policy_sha','split','price','cost','budget','tokens',
        'intervention','executed_action','tool_receipt','correct'}
ARMS={'skip','call','free'}

def admit(records):
    groups=defaultdict(lambda:defaultdict(list));seen=set()
    if not records:raise ValueError('empty')
    for r in records:
        if FIELDS-set(r):raise ValueError('missing fields')
        arm=r['arm']
        if arm not in ARMS:raise ValueError('unknown arm')
        if r['intervention']!=('none' if arm=='free' else 'action_intercept_only'):
            raise ValueError('prompt forcing not allowed')
        action=r['executed_action']
        if action not in ('call','skip'):raise ValueError('missing actual action')
        if arm!='free' and arm!=action:raise ValueError('forced action mismatch')
        receipt=r['tool_receipt']
        if action=='call' and (
          not isinstance(receipt,dict) or
          not all(receipt.get(x) for x in ('tool_id','trace_sha','output_sha'))
        ):raise ValueError('missing tool trace')
        if action=='skip' and receipt is not None:raise ValueError('spurious tool receipt')
        if r['split'] not in ('screen','confirm'):raise ValueError('invalid split')
        if type(r['seed'])!=int or type(r['correct'])!=int or r['correct'] not in (0,1):
            raise ValueError('invalid seed/outcome')
        for x in ('price','cost','budget','tokens'):
            z=r[x]
            if not isinstance(z,(int,float)) or not math.isfinite(z) or z<0:
                raise ValueError('invalid cost/usage')
        if r['tokens']>r['budget']:raise ValueError('over budget')
        if any(not isinstance(r[x],str) or len(r[x])<4 for x in
               ('model_sha','runner_sha','tool_sha','grader_sha','policy_sha')):
            raise ValueError('missing provenance')
        serialized=json.dumps(r['state'],sort_keys=True,ensure_ascii=False,
                              separators=(',',':')).encode('utf-8')
        state_hash=hashlib.sha256(serialized).hexdigest()
        key=(r['task'],state_hash,r['model_sha'],r['runner_sha'],
             r['tool_sha'],r['grader_sha'],r['policy_sha'],
             r['split'],r['price'],r['cost'],r['budget'])
        uid=(key,arm,r['seed'])
        if uid in seen:raise ValueError('duplicate seed/arm')
        seen.add(uid);groups[key][arm].append(r)
    for key,arms in groups.items():
        if set(arms)!=ARMS or len({len(v) for v in arms.values()})!=1:
            raise ValueError('unmatched action triplet')
    return groups

def certify(records,alpha=.05):
    if not 0<alpha<1:raise ValueError('alpha out of range')
    groups=[(k,v) for k,v in admit(records).items() if k[7]=='confirm']
    if not groups:raise ValueError('no confirm records')
    out=[];M=len(groups)
    for key,arms in groups:
        n=len(arms['call'])
        p1=sum(v['correct'] for v in arms['call'])/n
        p0=sum(v['correct'] for v in arms['skip'])/n
        q=sum(v['executed_action']=='skip' for v in arms['free'])/n
        eps=math.sqrt(math.log(3*M/alpha)/(2*n))
        lower_gain=p1-p0-key[9]-2*eps
        lower_skip=max(0,q-eps)
        out.append({'task':key[0],'n_each':n,
                    'certified':lower_gain>0 and lower_skip>0,
                    'regret_lower':max(0,lower_gain)*lower_skip})
    return out
