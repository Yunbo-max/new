# R006 — Four-branch counterfactual measurement audit

Date: 2026-10-24. Restore native tree revision 3 plus R004 and R005 reports; no default main direction. All B1/B2/B3/B4 remain active. No LLM inference, GPU, formal Work reviewer or native subagents were run. The 60-minute budget is an owner-proposed maximum, not actual reported runtime.

## B1 — Necessity is not uplift
When2Tool labels an item tool-necessary if the no-tool rollout fails. But the expected causal uplift is p1(h)-p0(h), not 1-p0(h). In an explicit counterexample p0=.4, p1=.3, no-tool failures occur 60% of the time even though tool use lowers average success by 0.1. Prior art: When2Tool (arXiv:2605.09252), Knowing-Doing Gap (2605.14038), AgentAbstain (NeurIPS 2026, arXiv:2607.10059, 263 paired act/abstain tasks). A basic act-versus-abstain study is not novel. Falsifier: benefit prediction adds no task-level gain beyond cost-calibrated necessity control.

## B2 — Positive-part single-rollout estimation bias
Let Y0~Bern(p0), Y1~Bern(p1) independently under a fixed pre-action history; cost 0<=c<=1. Then E[max(0,Y1-Y0-c)]=(1-c)p1(1-p0), whereas true actionable regret uses max(0,p1-p0-c). With p0=p1=.5 and c=0, the former is .25 although the latter is zero; Monte Carlo 250000 artificial pairs returns .24896. One-shot realized regret from R004/R005 must not be used as an unbiased model failure estimate. Use repeated forced continuations, uncertainty bounds, held-out policy evaluation, and compare CTA (2602.16699).

## B3 — Selection-only visible/hidden gap
Choose the best visible score among n candidates, each with independent Bernoulli(p) visible and hidden outcomes. E[visible winner]=1-(1-p)^n, E[hidden winner]=p, despite no candidate intentionally exploiting the tests. At n=10,p=.5 the gap is .4990234375. This stylized counterexample proves a visible/hidden gap alone cannot attribute intent. SpecBench (2605.21384), BAITBENCH (2608.30724) and Oct-2026 multi-agent transmission work (2610.12822) are close prior art. Falsifier: after controlling outer candidate selection, no inner-agent-specific departure remains.

## B4 — Budget vs spent-token selection
Under equal easy/hard difficulty and randomized budgets 1/3, mean success rises .55 to .775 (+.225), but tasks consuming 3 units have .60 success versus .68333 among tasks consuming only 1. The negative observed token association is confounded by difficulty even as budget helps every task. Compare randomized assigned budget ITT and established CATS/overthinking baselines, not post-treatment token correlations.

## Actual evidence / status
The local CPU suite `r006_counterexample_suite.py` passed 7 tests and emitted `R006_SYNTHETIC_RESULTS.json`; `r006_replay_contract.py` passed 7 separate tests of canonical context hashes and forced0/forced1/free triplets. These are NOT real LLM results. Full scripts and expanded records are in `R006_research_checkpoint.zip` attached to the conversation. No costs are reported as zero if unmeasured.

Independent real Work review is pending. Native tree has to be read and budget-checked before actual LLM jobs. Next wake must revisit all B1–B4 without automatic focus/park.