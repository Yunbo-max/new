# R008 — Evidence-grounded full-frontier research (24 Oct 2026)

**Restore:** original authoritative native `research-state/continuous/tree.json` revision 3, plus append-only R004–R007 reports. B1, B2, B3, B4 all remain independently **active**. No primary direction. `budget_verified=false`; Work subagents, GPU training, model inference, independent reviewer = **not run**. No background Work job and no claim of one-hour elapsed time.

## B1 — A published probe's stochastic training labels may need re-evaluation
Source audit: `Trustworthy-ML-Lab/when2tool` commit `8c00ef7b4f7576de5f00e7ed2d4bf99cdac7f30a`; inspected `src/extract_features.py` and `src/model.py`. The code forms `tool_necessary = 0 if nt_correct else 1`, from one no-tool evaluation. The vLLM sampling construction has no explicit source-level seed field (defaults may still be deterministic). A public reproduction at https://github.com/Trustworthy-ML-Lab/when2tool/issues/1 reports Qwen3-4B multi-hop AUROC 0.9257 versus paper 0.966, accuracy 0.8489 versus 0.947. An author confirmed single-run training labels and possible randomness, provided archived labels, and is investigating; **cause of the discrepancy is not established**.

For i.i.d. no-tool success Y~Bernoulli(p0), independently sampled labels L=1-Y disagree with probability `2p0(1-p0)`, reaching 0.5 at p0=0.5. Thus a single hard-no-tool label measures a sampled failure, not a stable probability or tool-benefit delta `p1-p0`. Falsifier: original labels stable in repeated seeded trials and probe generalization unchanged after uncertainty-aware targets. First pilot: 90 stratified single-hop + 30 multi-hop tasks, 5 explicit seeds, original model/settings; not yet run. Baseline: original Probe&Prefill and calibrated repeated-label majority. No novel method claimed.

## B2 — Certify positive counterfactual value without inferring motives
At a **genuinely matched decision state** h, define g=p1(h)-p0(h)-c. For n0/n1 independent bounded forced-action rollouts and M states, a simultaneous Hoeffding/Bonferroni radius for g is `sqrt(log(4M/delta)/(2n0)) + sqrt(log(4M/delta)/(2n1))`. Mark a *certified beneficial omission* only if the lower bound is >0 and the voluntary policy skips. A few forced samples are inconclusive, not evidence of rationality. State/policy identity must be independently checked; matching text hashes alone is insufficient. Subjective laziness or intention cannot be recovered from this. Baselines: Calibrate-Then-Act, CITA and CSI. Falsifier: calibrated baseline eliminates conditional suboptimal skips.

## B3 — Separate outer candidate choice from inner agent reward hacking
In a nested coding evaluator like SpecBench, selecting best visible among n independent Bernoulli(p) candidates yields expected visible success `1-(1-p)^n` and selected hidden success p, **without any strategic cheating**. Design fixed-candidate-pool outer-selection controls (random vs score-based), independently matched withheld tests, then factorially vary inner-agent proxy exposure. Check true semantic task violations with a sealed grader. Prior work: SpecBench, BAITBENCH, reward-hacking benchmark. Falsifier: the entire observable gap is explained by outer selection and test composition.

## B4 — Assigned budget is randomized treatment; consumed tokens a mediator
Estimate `E[Y|do(B=b)]` with randomly assigned budget B, not `E[Y|tokens-used=m]`. Changing max-token budget affects truncation, formatting, and stopping, so treating B as an instrument for 'pure reasoning' needs an exclusion assumption that often fails. Pilot budgets 128/256/512/1024, record actual consumed tokens, truncation and correctness, independent task/seed repeats. Compare standard adaptive compute controllers at matched resource costs. Falsifier: no reliable delayed/nonconcave effects or all improvements matched by existing baselines.

## Actual execution and receipts
The **locally generated R008 research package** contains `ROUND-008.md`, `r008_audit.py`, `test_r008_audit.py`, `README.md`. The stdlib audit handles repeated no-tool label stability and forced0/forced1/free conditional action values, with strict protocol checks and familywise intervals. **14/14 actual CPU unit tests passed**, using intentionally synthetic test arrays; **no real Qwen inference or model-performance estimate**. ZIP SHA256: `5748e555998fb7a8112fbbd0d9b7404f702412ff00401d32c3f79df680a09cab`. Full local report SHA256: `2cd45d51a06008b44b0b7072c63035846c27c3824898364772cebe24d0e965b2`.

## Next-science actors and gates
- Reviewer (separate genuine Work conversation): challenge B1 stochastic-label interpretation and estimand/continuation-policy validity, cross-check the author's issue and code; review B3/B4 basic confounds.
- Designer: retain all B1–B4 and write pinned model harness contracts with outcome-grader isolation.
- Local native executor, **only after explicit authorization**: validate a small repeated-seed When2Tool pilot before any new architecture/optimization claim.

This GitHub report is an **append-only scientific checkpoint**, not evidence of native tree revision advancement, formal review acceptance, model execution or automatic future wake.