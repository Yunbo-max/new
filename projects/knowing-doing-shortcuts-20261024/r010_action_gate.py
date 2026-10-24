"""R010: offline JSONL admission gate. Only stdlib; no model inference."""
import hashlib,json,math
from collections import defaultdict

FIELDS = {"task","state","arm","seed","model_sha","source_sha","mode",
          "cost","correct","tokens","budget","forcing"}
ARMS = {"skip","call","free"}

def audit(rows, confidence=.95):
    if not 0 < confidence < 1:
        raise ValueError("invalid confidence")
    grouped=defaultdict(lambda:defaultdict(list))
    seen=set()
    for row in rows:
        if FIELDS - row.keys():
            raise ValueError("missing fields")
        if row["arm"] not in ARMS or row["mode"]!="controlled_replay":
            raise ValueError("invalid arm or mode")
        expected="none" if row["arm"]=="free" else "action_only"
        if row["forcing"]!=expected:
            raise ValueError("prompt-forcing confound")
        if row["correct"] not in (0,1):
            raise ValueError("invalid outcome")
        if min(row["cost"],row["tokens"],row["budget"])<0:
            raise ValueError("negative resource")
        if not row["model_sha"] or not row["source_sha"]:
            raise ValueError("unpinned model/source")
        state_hash=hashlib.sha256(json.dumps(row["state"],sort_keys=True).encode()).hexdigest()
        key=(row["task"],state_hash,row["model_sha"],row["source_sha"],
             row["cost"],row["budget"])
        uid=(key,row["arm"],row["seed"])
        if uid in seen:
            raise ValueError("duplicate arm/seed")
        seen.add(uid)
        grouped[key][row["arm"]].append(row)
    if not grouped:
        raise ValueError("empty")
    report=[]
    M=len(grouped)
    for key,arms in grouped.items():
        if set(arms)!=ARMS:
            raise ValueError("missing counterfactual arm")
        ns=[len(arms[x]) for x in ("skip","call","free")]
        if len(set(ns))!=1:
            raise ValueError("unbalanced replay")
        n=ns[0]
        if any(r.get("action") not in ARMS-{"free"} for r in arms["free"]):
            raise ValueError("missing natural action")
        p0=sum(r["correct"] for r in arms["skip"])/n
        p1=sum(r["correct"] for r in arms["call"])/n
        skip=sum(r["action"]=="skip" for r in arms["free"])/n
        radius=math.sqrt(math.log(3*M/(1-confidence))/(2*n))
        regret_lower=max(0,skip-radius)*max(0,p1-p0-key[4]-2*radius)
        report.append({"task":key[0],"n_per_arm":n,
            "estimated_benefit":p1-p0-key[4],
            "skip_probability":skip,"regret_lower":regret_lower,
            "certified":regret_lower>0})
    return report

if __name__=="__main__":
    import sys
    data=[json.loads(s) for s in open(sys.argv[1],encoding="utf-8") if s.strip()]
    print(json.dumps(audit(data),indent=2))
