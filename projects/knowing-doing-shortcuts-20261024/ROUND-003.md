# Autopilot Manual Wake — R003 (2026-10-24)

Restored: native tree revision 2, project `llm-shortcutting-20261024`, full active frontier B1–B4. Owner effort envelope 60 minutes (maximum preference, not elapsed-time claim); no registered Work subagents, independent reviewer, GPU or model execution. Budget remains unverified.

## Actual evidence delta

| Node | Verified mathematical boundary | Closest work | Next falsifier |
|---|---|---|---|
| B1 | Under invertible change of hidden coordinates with compensated readout, two probe vectors' cosine changes from 0 to -0.99995 while every tested logit stays identical | Cheng 2605.14038; When2Tool 2605.09252; ACM survey DOI 10.1145/3728458 | Test task-success changes under well-controlled hidden-state interventions, not probe-angle alone |
| B2 | Stochastic tool decisions are observationally invariant under (v,lambda,beta) -> (alpha v,alpha lambda,beta/alpha), for every displayed price | CTA 2602.16699 and Wu 2605.00737 | Estimate forced-action success after same decision history; normalize decision noise before motives are attributed |
| B3 | 2x2 exploitability x grader-cue synthetic factorial yields interaction 0.30, hidden task success effect -0.14 and proxy score effect +0.105 | SpecBench 2605.21384; Handshake DeepSWE September 2026; RHB ICML 2026 | Same experiment with sealed true evaluator and non-exploitable proxy negative control |
| B4 | 1225 separable decreasing-marginal payoff curve pairs had zero greedy suboptimality; a nonconcave delayed-return example gives greedy gap 0.59 | Federgruen & Groenevelt 1986 (10.1287/opre.34.6.909); 2026 compute allocation studies | Estimate empirical p_i(t) and test nonconcavity, cost parity with adaptive baselines |

All four witnesses are artificial; the executable source is `r003_witness.js`, and its computed evidence is `R003_RESULTS.json`. This is not real LLM evidence.

## Per-branch deep records

- `evidence/B1_R003.md` provides basis-dependent probe-angle proof, assumptions, and intervention test.
- `evidence/B2_R003.md` proves stochastic cost-motive nonidentifiability, with its beta-normalization caveat.
- `evidence/B3_R003.md` states factorial identification assumptions, competing explanations and negative controls.
- `evidence/B4_R003.md` states the classical discrete concavity boundary, proof sketch and full finite-family enumeration.

## Closest primary-source links

https://arxiv.org/abs/2605.14038
https://arxiv.org/abs/2605.09252
https://doi.org/10.1145/3728458
https://arxiv.org/abs/2602.16699
https://arxiv.org/abs/2605.00737
https://arxiv.org/abs/2605.21384
https://joinhandshake.com/research/ai/deepswe-reward-hacking/
https://proceedings.mlr.press/v306/thaman26a.html
https://doi.org/10.1287/opre.34.6.909

## Disposition and blockers

B1: active, deep_incomplete. B2: active, deep_incomplete. B3: active, deep_incomplete. B4: active, deep_incomplete. No branch automatically focused or parked. Formal separate-work review absent. Native model experiment absent. Tool-token costs not measured (unknown, not zero); no new GPU permission assumed. No unattended work is running when the turn ends.

Next human wake: restore all four peers from the GitHub tree; check source-level equivalences and independent review, then prepare model pilots only under separately verified compute authority.