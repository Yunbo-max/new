#!/usr/bin/env python3
"""Four transparent synthetic witnesses, not LLM experimental results.
Run: python analytical_witness.py
"""
def b1_non_identifiability():
    k, a = (0, 1), (0, 0)
    probe_accuracy = sum(int(ki == gi) for ki, gi in zip(k, (0, 1))) / len(k)
    action_sensitivity = a[1] - a[0]
    assert probe_accuracy == 1.0 and action_sensitivity == 0
    return {"perfect_probe_accuracy": probe_accuracy,
            "action_change_when_knowledge_changes": action_sensitivity}

def b2_cost_belief_counterexample():
    p_skip, p_verify = .62, .82
    costs = (.01, .09)
    true_gain = [(p_verify - p_skip) - cost for cost in costs]
    subjective_gain_high = (.68 - p_skip) - costs[1]
    assert all(x > 0 for x in true_gain) and subjective_gain_high < 0
    return {"true_verify_gains_low_high": [round(x, 3) for x in true_gain],
            "subjective_gain_high": round(subjective_gain_high, 3)}

def b3_proxy_divergence():
    policies = {"honest": {"true": .82, "proxy": .80, "cost": .09},
                "shortcut": {"true": .42, "proxy": .97, "cost": .01}}
    true_argmax = max(policies, key=lambda a: policies[a]["true"] - policies[a]["cost"])
    proxy_argmax = max(policies, key=lambda a: policies[a]["proxy"] - policies[a]["cost"])
    assert true_argmax == "honest" and proxy_argmax == "shortcut"
    return {"true_objective_selects": true_argmax, "proxy_objective_selects": proxy_argmax}

def b4_effort_misallocation():
    easy, hard = (.93, .95, .93, .92, .90), (.30, .45, .63, .70, .71)
    choices = [(t, 4-t, easy[t] + hard[4-t]) for t in range(5)]
    optimum = max(choices, key=lambda x: x[-1])
    uniform = next(x for x in choices if x[:2] == (2, 2))
    assert optimum[:2] == (1, 3)
    assert round(optimum[2] - uniform[2], 8) == .09
    return {"optimal_allocation": optimum[:2],
            "optimal_expected_correct": round(optimum[2], 3),
            "uniform_expected_correct": round(uniform[2], 3),
            "gain": round(optimum[2] - uniform[2], 3)}

if __name__ == "__main__":
    for name, fn in [("B1", b1_non_identifiability), ("B2", b2_cost_belief_counterexample),
                     ("B3", b3_proxy_divergence), ("B4", b4_effort_misallocation)]:
        print(name, fn())
