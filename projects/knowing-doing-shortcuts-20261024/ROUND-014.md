# R014 — Evidence-conditioned actions: novelty audit and real observational evidence

Date 2026-10-24. Full-frontier Human Wake, B1–B4 all active, no global primary direction. Recover original native tree revision 3 together with append-only R004–R013. Previous project budgets remain `budget_verified=false`, no active issued wake. No native LLM/GPU experiment or separate independent reviewer.

## Major prior-art collisions

- Zhang et al., *Judged Useless, Queried Anyway* (2026-10-05), https://arxiv.org/abs/2610.06191 : agents judged failing retrieval results useless in 97–100% of recorded judgments but rarely stopped unless a harness enforced evidence integration. The paper reports a replication on 300 fresh questions; its episode dataset `benni23/judged-useless-episodes` is public. We inspected the primary paper and dataset metadata but did **not** download its episode rows. Recognition-not-action and forced integration are NOT a new R014 contribution.
- Yang et al., *AgentBoundary*, https://arxiv.org/abs/2609.33658 : 4-way executable counterfactual benchmark already separates action permission from risk appearance, and adds runtime action calibration. No novelty for a generic 'authorized action cue'.
- Lin et al., *SafeActBench*, https://arxiv.org/abs/2610.07753 : provenance-bound evidence ledger and action-state evaluator already separate investigation, unsupported action and workflow effects. Their reported conditional successful action rate is high when evidence is complete, but it does not resolve skipped-action counterfactuals.
- CATS https://research.google/pubs/cost-effective-agent-test-time-scaling-2/ and UK AISI https://www.aisi.gov.uk/blog/more-compute-more-capability-why-ai-agent-evals-need-to-account-for-test-time-compute already study budget curves and serial/parallel allocation.

## B1 — Observed evidence and residual missed action
**Operational boundary:** pre-action records with provenance, entity ID, freshness, permission and tool affordances. 'Evidence observed in history' does not establish the LLM's private belief or knowledge. A missed-benefit claim requires a skipped *permitted* action after complete evidence and a separately certified positive counterfactual value. `Judged Useless` and SafeAct already overlap the high-level mechanism.
**Counterexample / rival:** a model can explicitly label evidence useless and still query for hidden valid reasons (alternative information or missing authorization); naive interpretation confuses evidence judgment with task action value.
**Simple baseline and falsifier:** author evidence-integration/ledger plus cost-calibrated action threshold; if residual beneficial omissions disappear after these standard methods at matched information and cost, abandon novelty claim. No model experiment performed.

## B2 — Separate evidence collection, action policy and world effect
**Mechanism and code:** `r014_preflight.py` checks chronological, entity-correct, source-verified observations, permission, action timing, tool state effect, and legitimate stop. It distinguishes unsupported/unauthorized action, fake successful no-op, wrong-state mutation, correct abstention, evidence-complete omission with confirmed positive value, and unknown-value omission. It refuses post-stop evidence and hidden gold field leakage.
**Exact limitation:** externally supplied `sealed_positive_value` is an off-policy evidence requirement, not a probability inferred by the code; hashes and tool receipts do not authenticate model states. Existing SafeAct/evaluation contracts already address much of this; the code is a sanity gate, not novel agent policy.
**Falsifier:** after authentic state-level checking and strong baseline calibration there is no verified action policy gap. Next actor: separate reviewer before native model integration.

## B3 — Outer candidate selection versus inner LLM strategy
**Primary-code inspection:** `WecoAI/SpecBench` `benchmarks/spec_bench/run_loop.py` (blob `2f15055a639e9de849821c926b5b5df7de14a851`) passes **public evaluation logs** to AIDE's `update_latest_candidate`, stores private scores in a separate result, then selects `get_best_node()`. An apparent visible/hidden gap need not imply inner strategic cheating. A fixed-pool ranker is a simple negative-control baseline.
**Falsifier / task:** freeze candidates, randomize outer ranking; separately randomize inner proxy visibility with fixed outer policy and a sealed hidden evaluator. No SpecBench native runs have been executed in R014.

## B4 — Termination is a confound in real coding-agent logs
**New direct real-data aggregate:** source `latitude-dev/coding-agent-benchmark/results/results.jsonl` GitHub blob `c8dadddf750e9068ca4aa5aa53935bc799db5792`, 847 rows. 817 normal `stop` (727 successful, 90 failed, mean output tokens 1343.73); 29 `content-filter` (2 successful, 27 failed, mean 191.48); 1 `length` (failed, 4331 tokens). All 29 filtered runs are from model `claude-fable-5`. Pooled failed-minus-solved output tokens +1188.72; in normal-stop subset it is +1816.82. These are observational source aggregates; cannot identify causal compute effects or laziness.
**Falsifier / task:** exogenously randomize assigned inner and outer compute budgets within matched model/task strata. Analyze all termination reasons including safety filters, truncation, consumed tokens and post-state success, with CATS/Bandit/standard fixed policies as strong controls.

## Execution receipts and continuity

- Full local directory in `R014_research_checkpoint.zip` includes detailed `ROUND-014.md`, `R014_OBSERVATIONS.json`, `r014_preflight.py`, `r014_budget_audit.py`, tests, and a 86-case SafeAct Legacy compact *gold-only* CSV from the previous real-source audit.
- **28/28 local Python stdlib tests passed**. This includes actual parsing of 86 real benchmark gold labels, and synthetic event-level tests. The 847-run aggregate was computed from connected pinned GitHub data; its raw JSONL was not copied to the local container, so the script requires user-supplied raw file to reproduce independently.
- No LLM inference, provider/GPU cost authorization, formal independent reviewer, model-reported success, or background job. The native tree is unchanged revision 3 until separately reconciled. The next human wake must restore B1–B4 and all R004–R014 reports.
