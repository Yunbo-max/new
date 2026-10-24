# R013 — Real SafeAct source audit and causal admission gates

Date: 2026-10-24. Continue full B1–B4 frontier from native tree revision 3 and append-only R004–R012 reports. No default global main direction; existing budget authority remains unverified. No LLM/GPU model trials or independent real Work reviewer were executed.

## Direct real-data analysis

Inspected entire **86-case Legacy subset** in `caoshidong66/safeact/env/cases.json`, pinned Git blob `baee3298bb851733b50b862526ba2e39fe8fd0ad`. This is a subset of 656 benchmark tasks, not all of SafeActBench. Ground-truth decisions: allow 29, defer 34, block 23. 66 cases contain hidden `gold_confirm.status` and 20 have a different field layout. In those 66, the gold confirmation status maps deterministically to the gold decision (66/66): satisfied -> allow; failed -> block; missing, stale, conflicting -> defer. **This is evaluator-only label information and must NOT be included in agent-facing inputs or as a model knowledge feature. The published benchmark hides its gold evaluator; no official leakage is alleged.**

There are 53 explicit contrast group IDs. Nine contain multiple cases; seven change the gold decision. All nine multi-case families differ in candidate tool/arguments and in user request. Hence benchmark contrast ID does not certify fixed pre-action state or a single randomized treatment for a causal replay. Declared expected evidence tools are subsets of declared available tools in inspected cases, but availability is not observed evidence acquisition or successful execution.

The SafeActBench authors report conditional single-action success 93.2–100% for nine of ten model-harness configurations **once evidence is complete and an action is taken**. This does not identify the counterfactual success of agents that would have skipped.

## Full-frontier peer outcomes

**B1 – Decodable knowing vs. actual pre-action evidence.** A knowledge probe cannot read hidden gold annotations at test time. A sound missed-benefit claim requires evidence actually observed *before* an available positive-value action that was skipped. Conditioning on the subset who acted may induce selection bias. Closest prior work: SafeActBench, When2Tool, AgentAbstain, CITA, CSI. A standard evidence-ledger preflight is the simplest repair. **Falsifier:** no residual missed beneficial actions beyond preflight and properly calibrated tool-value choices.

**B2 – Real state effect vs. success receipt.** R012's SQLite no-op/wrong-target cases show raw tool receipt is insufficient. Legacy SafeAct gold decisions are static; they do not reveal forced-call vs forced-skip world effects. Move to V1+ stateful tool evaluator and causal metric E[Y(W_after_call)-Y(W_after_skip)|h]-cost. Compare the official SafeAct ledger and executable contract checking. **Falsifier:** state-level verification alone eliminates the purported action gap.

**B3 – Inner agent changes vs outer score selection.** Source inspection confirms SpecBench `benchmarks/spec_bench/run_loop.py` uses public evaluation to update AIDE candidate search, while recording private scores separately. Fixed-pool outer ranking and inner proxy-incentive exposure must be varied orthogonally; SafeAct's nine Legacy contrast families are not matched action states. **Falsifier:** selection-only null reproduces the observed public/private performance difference.

**B4 – Assigned budget vs. consumed tokens.** SpecBench has outer cost/time/step limits; ICML 2026 Conformal Thinking, COLM CATS and adaptive/bandit test-time budgets are strong existing baselines. Randomize outer search budget and inner generation cap separately before the run; record actual token consumption as a possible mediator, and grade trusted world states. **Falsifier:** cost-matched standard policies explain all gains and no reproducible delayed benefit remains.

## Reproducibility and honest state

Portable local bundle `R013_research_checkpoint.zip` contains `ROUND-013.md`, derived evaluator-only 86-case CSV, `audit_legacy.py`, `audit_source_contract.py`, tests and source-audit results. Package SHA256 `a7f18a733a8c1fefa94a5bddf5ddaf3cf5b47485b7e14b5efa97e6d70e01ec9a`; **13/13 real-summary/schema+negative-control CPU tests passed**. Connected GitHub analysis parsed the full 86-case author JSON; the original full source was not locally materialized due to disabled container networking, and `audit_source_contract.py` can recheck it when supplied externally. No real Qwen inference, no independent formal review or GPU job. Provider cost unmeasured, not zero.

**Next wake:** all four active nodes must be restored; inspect SafeAct V1 agent-visible evidence and tool state, independently review causal protocol, then use authorized native agent rollouts to test genuinely matched intervention decisions. The original native tree remains revision 3 until guarded reconciliation; this report is append-only, not a fabricated controller batch.

Sources: https://github.com/caoshidong66/safeact ; https://safeact.github.io/ ; https://arxiv.org/abs/2610.07753 ; https://github.com/WecoAI/SpecBench ; https://proceedings.mlr.press/v306/wang26hd.html .
