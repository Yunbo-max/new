"""R006 synthetic-only sanity checks. Run: python r006_math_witness.py"""
from math import isclose
import json

def run():
    # B1: failure without tool is not a positive tool benefit.
    p0,p1=.4,.3
    b1={"no_tool_failure_rate":1-p0,"tool_uplift":p1-p0}
    assert b1["tool_uplift"]<0

    # B2: positive-part of one-shot outcomes is optimistically biased.
    p0=p1=.5
    naive=p1*(1-p0)
    b2={"one_shot_expected_positive_part":naive,"true_value":max(0,p1-p0)}
    assert b2["true_value"]==0 and naive==.25

    # B3: picking max visible score creates a gap without agent strategy.
    n=10;p=.5
    b3={"expected_winner_visible":1-(1-p)**n,"expected_winner_hidden":p}
    assert isclose(b3["expected_winner_visible"]-p,.5-2**-10)

    # B4: assigned budget effect can oppose observed token association.
    b1_mean=(.9+.2)/2
    b3_mean=(.95+.6)/2
    tok1_mean=(.9+.2+.95)/3
    b4={"budget_gain":b3_mean-b1_mean,"tokens_3_minus_1":.6-tok1_mean}
    assert b4["budget_gain"]>0 and b4["tokens_3_minus_1"]<0
    return {"kind":"synthetic_not_llm","B1":b1,"B2":b2,"B3":b3,"B4":b4}
if __name__=="__main__":
    print(json.dumps(run(),indent=2))
