# R007 - Causal value validation, genuine novelty collisions

Date: 2026-10-24. Restored authoritative tree revision 3 plus append-only R004/R005/R006. Four active peers B1–B4; no primary direction or Work subagents. Budget not verified; no model/GPU experiment or independent reviewer.

## Highest-impact source findings

1. CITA (NeurIPS 2026; arXiv:2610.02330) already learns long-horizon *comparative* values of alternative tool actions under similar or identical contexts using observed trajectories, Bayesian tool-graph simulation, and LLM comparisons. Relative tool-value estimation by itself is NOT new.
2. CSI (Findings EMNLP 2026; https://aclanthology.org/2026.findings-emnlp.1532/) already uses *same-state counterfactual returns* to selectively train structured agent action spans. Selective action improvement is NOT new.
3. Counterfactual Tool Ranking (arXiv:2609.22819) already examines cost, action support and conservative bounds, so uncertainty-based action selection is NOT independently novel.
4. Existing reference benchmarks: When2Tool, CTA, SpecBench, BAITBENCH, AgentAbstain.

## Each branch's new analytical boundary

**B1:** Even matching the pre-action prefix does not identify the tool's causal effect if forced action prompts alter later continuation. Artificial witness: natural call/skip outcomes both .5 (effect 0), forced-protocol call=.9 and skip=.1 (spurious effect .8). Fix the subsequent policy; compare against When2Tool, CITA and CSI.

**B2:** Let Q1,Q0 be average bounded task outcomes for forced-call and forced-skip using n independent continuations per arm from the same state. A conventional Hoeffding 1-delta lower bound on Q1-Q0-c is (Qhat1-Qhat0-c)-2sqrt(log(4/delta)/(2n)). This is a *diagnostic*, not a new method. In 100 seeded artificial null histories (n=128 each arm), point estimates chose the tool 39 times; certified lower bounds chose 0. In 100 large-gain examples the lower bound chose 100. Evaluate CITA, CSI, CTA and existing counterfactual tool-ranking/uncertainty baselines before pursuing a method.

**B3:** Selecting the highest *visible* score among 10 fixed independent candidate solutions creates an apparent visible-hidden gap even with no strategic reward hacking. Across 5000 artificial replicates: winner visible .9988, hidden .4968; fixed candidate visible .4922, hidden .4924. Separate outer candidate selection from inner model generation and evaluate on a sealed independent set; compare SpecBench and BAITBENCH.

**B4:** Randomized assigned budget can increase success while observed consumed-token correlations are negative because harder tasks consume more. Artificial expected success by assigned budget 1 vs 3: .55 vs .775, yet consumed-token groups 1 vs 3: .6833 vs .60. Compare assigned-budget ITT to adaptive compute baselines CATS, MUTO and OptimalThinkingBench.

## Actual work and gates

A complete CPU-only standard-library artifact `R007_research_checkpoint.zip` was produced in this conversation: `r007_certificates.py` (9 automated assertions), `R007_SYNTHETIC_RESULTS.json`, full `ROUND-007.md` with exact assumptions and per-leaf D1–D6, and README. **All numbers artificial.** No native Qwen, no Work reviewers or GPU run, unknown provider token costs, no authorized compute increase. Candidate real pilot after independent review: Qwen3-1.7B/Qwen3-4B-Instruct-2507, action-only controlled replay with identical continuing policy and sealed scoring; 20-50 tasks and 8 seeds/arm for initial variance inspection only.

Four leaves remain active, all `deep_incomplete`. Next human wake must check every peer, exact report, reviewer status, budget and preserved native tree; do not treat the append-only record as an updated native tree revision.
