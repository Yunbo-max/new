# R010 — Scientific checkpoint: information boundary and counterfactual action value

Date: 2026-10-24. Source: native revision 3 + R004–R009 append-only deltas. Active nodes B1/B2/B3/B4; no global main direction. Budget remains unverified. No Work subagents, native model/GPU run, or independent reviewer.

## B1 — Label uncertainty and an independent reproduction
Official When2Tool code `src/extract_features.py` labels tool necessity with `tool_necessary = 0 if nt_correct else 1` after a `hard_no_tool` rollout, which is not the same as counterfactual positive tool value. Its public GitHub issue #1 remains open; a reproducer reports multi-hop Qwen3-4B AUROC 0.9257 / accuracy 0.8489 vs paper 0.9658 / 0.9467; an author confirmed single-run train labels and offered their original archive, without identifying the root cause.

A separate third-party model card `Dwootton/when2tool-tool-intent` (updated 7 Oct 2026) reports AUROC 0.9445 and accuracy 0.9489 when using author's labels. This third-party reproduction was NOT run or independently validated by us; aggregate metrics cannot uniquely establish label noise as the source. Source: https://huggingface.co/Dwootton/when2tool-tool-intent ; https://github.com/Trustworthy-ML-Lab/when2tool/issues/1 .

Formal identity: independent no-tool outputs with task success p produce sampled-necessity label disagreements with probability 2p(1-p). Next falsifier: get original archive + per-task scores and test frozen probe with repeated no-tool labels. Author's binary attachment could not be downloaded in this environment.

## B2 — Failure recovery can be an information-boundary problem
The third-party Qwen3-4B model card reports 63 controlled fork origins per condition with success: normal .49, persistent denial .25, denied + explained alternative .78, alternative absent .24. Its report describes persistence after tool denial and rapid uptake when the *name of the alternative capability* is revealed. These are third-party values, not this round's model runs. Text hinting changes accessible information; one cannot infer intentional laziness or 'knowing but refusing' from this contrast.

Next falsifier: make a legitimate alternative available, factorially randomize whether it is documented before failure and whether the original tool is transiently or persistently denied. Fix original state and continuation policy except the declared interventions. Measure actual switches, retries and hidden correctness. If documented alternatives eliminate persistence, classify as an information/affordance gap. Baselines: CTA, CITA, CSI and ordinary replanning.

## B3 — Published reward-hacking methods overlap our original idea
Yang et al. arXiv:2609.04665, HackProbe, already propose black-box proxy-gap detection with a secret comparison core and honest candidate reselection; Goodfire Sep-2026 reports activation monitors during reward hacking. Therefore an evaluator-swap, secret held-out tests or basic honest reselection are insufficient novelty.

Mechanism test: fix the generated candidate program pool and randomize only outer selector (random vs visible-score ranking). Separately randomize inner-agent exposure to proxy incentives with the outer selector held fixed, keeping actual task definition and hidden checker sealed. Refute B3 if a selection-only null and task-mix shifts explain all apparent reward hacking, or if exposure doesn't affect inner policy.

## B4 — Adaptive computation has mature baselines
ICLR 2026 `Strategic Scaling of Test-Time Compute: A Bandit Learning Approach` allocates budget by learnable query difficulty, including hard and unsolvable cases, with real code/math experiments. COLM 2026 CATS allocates token and tool resources across search-agent exploration. UK AISI finds fixed compute caps materially affect model capability. Naive 'more effort when needed' is not novel.

Test fixed model and task splits, exogenously randomize budget and tool-call allowance, collect repeated success curves, measure actual tokens and distinguish serial from parallel budget. Strong baselines: CATS, bandit allocation, calibrated stopping and cost-matched fixed policies. Refute B4 if no statistically reliable delayed/nonconcave gain remains.

## Executed artifacts and verification
- Offline admission tool `action_value_gate.py`: checks exact recorded pre-action state, pinned model/source revisions, forced-call/skip/free arm integrity, prompt-forcing contamination, duplicates; computes simultaneous conservative confidence bounds on skipped tool value, and repeated label discrepancy summaries. This checks metadata; it CANNOT verify true model state equality, independent sampling or grader isolation.
- `test_action_value_gate.py`: 11/11 stdlib unit tests passed on synthetic records in this finite R010 run, NOT real Qwen experiment.
- Full local ZIP: `R010_research_checkpoint.zip`, SHA256 `9da555ee769bb49d4cee0000744d8651c05b60acbe51ecd02b353b599c074385`.
- Unmeasured provider cost unknown, NOT zero. No model inference, independent Work review or unapproved GPU job.
- Next wake: reconcile full four-node R010 frontier and original native revision 3; obtain actual author original label archive, request independent reviewer, then separately authorize real model pilot.

Sources: https://huggingface.co/Dwootton/when2tool-tool-intent ; https://arxiv.org/abs/2609.04665 ; https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8d2c5f79c53a2a685dbdf93f78c5695-Abstract-Conference.html ; https://research.google/pubs/cost-effective-agent-test-time-scaling/ .
