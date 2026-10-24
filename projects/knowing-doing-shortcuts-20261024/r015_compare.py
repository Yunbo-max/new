"""R015: compare authors' actual model results (standard library only).
Run: python r015_compare.py <path-to-author-paper_values.json>
Source: bennidict23/judged-useless-queried-anyway,
experiments/episodes/paper_values.json, blob a248f629a30e4ae083a4137a3e01a257d4f03f28.
This code runs NO language model and cannot produce per-question CIs.
"""
import json,sys
models=["qwen2.5-7b","llama3.1-8b","qwen3-8b","qwen3-32b"]
splits=["test300","fresh300"]
def compare(p):
    rows=[]
    for split in splits:
        for model in models:
            def arm(name):
                x=p[f"{split}/{model}__{name}"]
                assert x["ci_low"]<=x["delta"]<=x["ci_high"]
                return x
            b,e,n=arm("budget"),arm("enforced_rule"),arm("unaided")
            rows.append(dict(split=split,model=model,
                delta_change=e["delta"]-b["delta"],
                mean6_change=e["mean6"]-b["mean6"],
                budget_gain=b["mean6"]-n["mean6"]))
    assert len(rows)==8
    return dict(rows=rows,
        delta_improves=sum(x["delta_change"]>0 for x in rows),
        success_improves=sum(x["mean6_change"]>0 for x in rows),
        success_worsens=sum(x["mean6_change"]<0 for x in rows),
        note="Published author aggregates, NOT new inference; paired episode CI unavailable.")
if __name__=="__main__":
    print(json.dumps(compare(json.load(open(sys.argv[1]))),indent=2))
