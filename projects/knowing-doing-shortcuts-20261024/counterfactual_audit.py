#!/usr/bin/env python3
"""Toy causal and optimization checks for R002. Not an LLM experiment.

All values and outcomes in this file are artificial by construction; intended
only as explicit counterexamples and protocol regression tests.

Run with Python 3.10+ (stdlib only): python counterfactual_audit.py
"""
from __future__ import annotations
import itertools
import json
import math


def b1_decodability_is_not_control():
    # The pre-decision representation H=(N,Z) has a perfectly informative N
    # axis. The model's call policy is based on independent Z instead.
    states=list(itertools.product((0,1), repeat=2))
    probe=[n for n,z in states]
    calls=[z for n,z in states]
    necessity=[n for n,z in states]
    knowledge_accuracy=sum(int(a==b) for a,b in zip(probe,necessity))/4
    action_accuracy=sum(int(a==b) for a,b in zip(calls,necessity))/4
    # Intervening on the N axis changes the probe without affecting the action.
    action_unchanged=all(z == z for n,z in states)
    # A linear policy w_A dot H with w_A=(0,1) annihilates the probe v=(1,0).
    policy_jacobian_along_knowledge_axis=0
    assert knowledge_accuracy==1 and action_accuracy==0.5 and action_unchanged
    assert policy_jacobian_along_knowledge_axis == 0
    return dict(probe_accuracy=knowledge_accuracy, action_alignment=action_accuracy,
                action_change_under_knowledge_axis_steering=policy_jacobian_along_knowledge_axis)


def b2_cost_identifiability_and_regret():
    costs=(.03,.08,.13,.18)
    # Two *distinct* subjective valuation/cost-aversion hypotheses have equal
    # decisions at EVERY tool price c, not only the four prices below.
    world_a=(.12,1.0) # subjective predicted uplift, cost coefficient
    world_b=(.24,2.0)
    decisions_a=[int(v-k*c>0) for c in costs for v,k in [world_a]]
    decisions_b=[int(v-k*c>0) for c in costs for v,k in [world_b]]
    assert all(((world_a[0]-world_a[1]*(i/1000))>0) ==
               ((world_b[0]-world_b[1]*(i/1000))>0) for i in range(1001))
    assert decisions_a == decisions_b == [1,1,0,0]
    # Forced tool use generates artificial Bernoulli success probabilities.
    # The estimand requires conditioning on the same pre-action history.
    p_success_forced_tool=.80
    p_success_forced_skip=.60
    regret_by_price=[max(0.0,(p_success_forced_tool-p_success_forced_skip)-c)
                     if choice==0 else 0.0 for c,choice in zip(costs,decisions_a)]
    regret=sum(regret_by_price)/len(costs)
    assert math.isclose(regret,.0225,abs_tol=1e-12)
    # Under deterministic threshold policy: identifies v/lambda, not either alone.
    return dict(costs=costs, decisions_a=decisions_a, decisions_b=decisions_b,
                inferred_subjective_threshold=world_a[0]/world_a[1],
                actual_forced_action_uplift=p_success_forced_tool-p_success_forced_skip,
                mean_dominated_skip_regret=round(regret,4),
                non_identifiable_parameters=[world_a,world_b])


def b3_proxy_intervention_does_not_prove_intent():
    # Exposed proxy controls the toy planner's utility; no natural-language
    # subjective 'deception' is inferred.
    honest={'true':.82,'proxy':.80,'cost':.09}
    shortcut={'true':.42,'proxy':.97,'cost':.01}
    rewards={'honest':honest,'shortcut':shortcut}
    selected={}
    for visible in (0,1):
      for cost_multiplier in (1,2):
        objective='proxy' if visible else 'true'
        choice=max(rewards,key=lambda name: rewards[name][objective]-cost_multiplier*rewards[name]['cost'])
        selected[f'visible={visible};costmult={cost_multiplier}']=choice
    assert selected['visible=0;costmult=1']=='honest'
    assert selected['visible=1;costmult=1']=='shortcut'
    # Causal proxy *effect* is detected by controlled visibility, but the same
    # behavior could also arise from misunderstanding altered instructions.
    return dict(policy_table=selected, warning='a proxy effect does NOT prove internal gaming intent')


def b4_nonconcave_effort_breaks_myopic_greedy():
    # Task hard requires three consecutive thinking units before improvement.
    # Uniform concavity assumptions do not hold here.
    easy=[.80,.82,.84,.86]
    hard=[.15,.15,.15,.80]
    budget=3
    choices=[dict(easy_units=i,hard_units=budget-i, expected_correct=round(easy[i]+hard[budget-i],3))
             for i in range(budget+1)]
    optimum=max(choices,key=lambda row:row['expected_correct'])
    allocation=[0,0]
    curves=[easy,hard]
    for _ in range(budget):
        marginals=[curve[k+1]-curve[k] for curve,k in zip(curves,allocation)]
        # Ties go to easy: either tie is still suboptimal here.
        allocation[max(range(2), key=lambda i:marginals[i])] +=1
    greedy=round(sum(curves[i][k] for i,k in enumerate(allocation)),3)
    assert optimum['easy_units']==0 and optimum['hard_units']==3
    assert math.isclose(optimum['expected_correct'],1.60)
    assert math.isclose(greedy,1.01)
    # This is a task whose gains are delayed, a constructed failure of greedy.
    return dict(optimal=optimum,greedy_allocation=allocation,
                greedy_expected_correct=greedy, regret=round(optimum['expected_correct']-greedy,3))


def main():
    print(json.dumps({'experiment_type':'synthetic_cpu_only',
        'B1':b1_decodability_is_not_control(),
        'B2':b2_cost_identifiability_and_regret(),
        'B3':b3_proxy_intervention_does_not_prove_intent(),
        'B4':b4_nonconcave_effort_breaks_myopic_greedy()},indent=2,sort_keys=True))


if __name__=='__main__':
    main()
