# R009 — Full-Frontier Research: Measurement and Calibration

Date: 2026-10-24. Restore native tree revision 3 plus independent R004–R008 delta/report records. All B1–B4 remain active without a primary direction. Native `budget_verified=false`, no Work subagent, reviewer acceptance or LLM/GPU inference.

## B1 — When2Tool reproduction diagnosis
Original code labels tool necessity by one `hard_no_tool` evaluation and not by the causal benefit of tool use. Public GitHub issue #1 (https://github.com/Trustworthy-ML-Lab/when2tool/issues/1) remains open; reported multi-hop Qwen3-4B AUROC 0.9257 vs published 0.966, accuracy 0.8489 vs 0.947. Author acknowledged train-set labeling was a single inference run and shared original probe data; no confirmed root cause. For independent runs with no-tool success probability p, binary label discordance is 2p(1-p). Five samples on a single task have two-sided 95% Hoeffding radius 0.607; to target +/-0.1 on one prespecified task requires 185 samples under that bound. **Thus 5 seeds are only a screening tool for aggregated task-level instability, not a precise per-task probability certificate.** The local R009 source implements task-bootstrap disagreement and a separate official-label versus re-run label/frozen-probe comparison. Author archive contents were not retrieved or inspected here. Next falsifier: stable author-vs-repeat labels and no probe metric movement under a frozen-score comparison.

## B2 — Certify skipped *policy* regret, not just tool benefit
For a fixed pre-action history h, let g=p1-p0-c and q=P(voluntary skip|h). The conditional actual objective is q*max(0,g), not max(0,g) alone. For M prespecified states and n0,n1,nfree independent samples, simultaneous Hoeffding margins eps_j=sqrt(log(6M/delta)/(2n_j)) give lower bound max(0,qhat-eps_free)*max(0,p1hat-p0hat-c-eps1-eps0), subject to proper matched policies/states, reward/cost normalization and independent draws. Selecting candidates on the same data invalidates naive confirmatory interpretations. A broad low-cost screening subset followed by held-out repeated confirmation is necessary. The local R009 audit tests 5-versus-500 sample precision. Compare original When2Tool, CTA, CITA, CSI and basic calibrated thresholds. Falsifier: no significant regret remains under strongest simple baselines at equal cost.

## B3 — Outer selection can produce a spurious hidden gap
Among n independent Bernoulli(p) candidate visible results, max visible has expectation 1-(1-p)^n while the selected candidate hidden result remains p when independent. This is a stylized selection null, not a model of SpecBench actual performance. For real studies, freeze the candidate program pool, vary only the outer score selector (random versus visible-rank), then fix outer selector and vary inner-agent feedback, preserving task and hidden evaluator. Compare SpecBench, BAITBENCH and RHB. Falsifier: the whole observed phenomenon is explained by the outer selector and test difficulty composition.

## B4 — Assigned budget versus post-treatment token use
Randomize assigned generation budget B before the episode, measure actual token consumption M and correctness Y, then estimate conditional intention-to-treat E[Y|do(B=large),stratum]-E[Y|do(B=small),stratum]. Conditioning on realized M may induce bias; changing B might also change truncation and output policy, so this is not isolated 'one extra reasoning token'. The local R009 analyzer checks strata and budget arms. Compare strong adaptive compute baselines. Falsifier: no statistically reliable nonconcavity/delayed value or a known method recovers all improvements.

## Actual receipts
- Re-ran R008 audit: 14/14 CPU tests passed.
- R009 code: `r009_methodology.py`, `reference_label_audit.py`, two test modules, complete `ROUND-009.md` and `R009_CPU_RESULTS.json` in the conversation's R009 ZIP; 21/21 CPU tests passed. No model experiments.
- Synthetic example: 100 tasks x 5 repeats yields disagreement 0.30 (task-bootstrap 0.24–0.36); these are fabricated fixture labels, not Qwen.
- ZIP SHA256: `5f1c989a7ef5f6caf1b308705f9e793fc174dc938c7a98fa3fe789068738a7e5`.

## Handoff and blockers
No genuine independent reviewer has accepted the equations, tests or native experiment design. Author-provided binary archive was not inspected; Qwen repeated inference awaits separate model/compute authorization and reviewed evaluation protocol. Current scientific level: `deep_incomplete` for all B1–B4. Native tree remains revision 3; this report is **append-only**, not a fabricated issued batch or automatic scheduled wake. Next manual wake must inspect all four independent leaves and the complete R009 report.
