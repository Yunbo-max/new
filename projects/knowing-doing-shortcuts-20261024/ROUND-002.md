# R002 - Causal identifiability, prior-art collisions, and action-level falsifiers
Date: 2026-10-24. Research mode: branching, no primary direction. Human-woken continuation of R001, under a proposed 60-minute maximum; actual elapsed time/provider usage unknown.

## Receipt and authority
Restored native `persistent-research-tree-v2` revision 1 from `Yunbo-max/new` branch `research/known-doing-shortcuts-20261024`. Four peer active leaves B1-B4 were intact; `budget_verified=false`, `wake=null`, with no native GPU/model job, reviewer or registered subagent dispatch. No existing scientific project outside this scope was imported. This run performed public primary-source checking, independent mathematical reasoning by the assistant (not an independent reviewer), and execution of artificial Python CPU witnesses. There are no claims of real LLM behavioral measurements.

## B1: Information decodability is not policy usage
**Nearest original methods:**
- Cheng et al. (2026), https://arxiv.org/abs/2605.14038 : model-dependent tool necessity; large cognition/execution gap and late representation decoupling.
- Sun et al., NeurIPS 2026, When2Tool: https://arxiv.org/abs/2605.09252 ; project https://lilywenglab.github.io/when2tool/ ; code https://github.com/Trustworthy-ML-Lab/when2tool. Pre-generation probe of tool necessity (AUROC 0.89-0.96); Probe&Prefill reduces calls. Inspected actual `src/train_probe.py` (logistic regression on hidden states) and `src/run_probe_eval.py` (precomputed hidden states and soft/hard prefill).
- Chen et al. (2026), https://arxiv.org/abs/2608.25198 : a direction steers tool call propensity from about 0% to >90%; a direction-based controller itself is NOT new.
- Park et al., ICML 2026, https://proceedings.mlr.press/v306/park26v.html : knowledge/prediction subspace alignment with an intervention.

**Formal witness.** Let latent N in {0,1} be necessity and independent Z in {0,1} be an action prior, both equally likely. Hidden state H=(N,Z). The linear knowledge probe k(H)=H_1=N is perfect (accuracy 1), while policy A(H)=H_2=Z agrees with necessity half the time. The linear action readout has W_A=(0,1), and the probe direction v=(1,0) lies in its kernel: W_A v=0. Steering only along v cannot change the call logit under this constructed policy. This establishes an exact existence counterexample, not empirical proof that an LLM discards knowledge.

**Rival and cheap repair.** The label N may represent a statistical property not causally accessible at decision time, or the agent may optimize real cost and correctly skip. Readout patching, supervised probe prefill, or a simple calibrated tool-choice controller are already powerful baselines; neither a new probe nor generic steering is novel.

**Discriminating test.** For the SAME pre-action history h: (1) estimate the probe out of distribution; (2) calculate its direction, actual decoder action direction and interaction; (3) randomized low/high costs with identical authorized tools; (4) intervene at specified layers on necessity-direction versus action-direction with matched norm; (5) score actual hidden task success, not just call rate. The branch's specific interaction hypothesis is refuted if necessity-direction manipulation has no selective beneficial task effect, or if existing Probe&Prefill/steering achieve the same conditional improvement.

**Status:** new source-and-code equivalence; a CPU logic witness; `deep_incomplete`, no independent reviewer.

## B2: Subjective cost-aversion is not identifiable from revealed actions
**Nearest original methods:**
- Ding et al. (2026), Calibrate-Then-Act: https://arxiv.org/abs/2602.16699 ; code https://github.com/NSF-Simons-CosmicAI-Institute/CalibrateThenAct . Its published tasks already include coding with selective unit testing and explicit prior calibration. Inspected `env_explorer/scripts/run_eval_code.py`, including `--calibrator` and `--rho`.
- Wu et al. (2026), To Call or Not to Call: https://arxiv.org/abs/2605.00737 . Already separates necessity, utility, and affordability and estimates model-perceived versus normative need; a 'need vs. benefit' controller is NOT a standalone novel idea.
- Sun et al. When2Tool and Chen et al. steering provide additional action-gating baselines.

**Proposition (exact under a deterministic threshold model).** Let the model choose verify iff `v - lambda*c > 0`, where `v>0` is subjective anticipated benefit, `lambda>0` is cost sensitivity, and randomized tool price c is externally visible. For any alpha>0, the pair `(alpha*v, alpha*lambda)` induces the identical action at EVERY c, because multiplying a real number by positive alpha preserves its sign. Consequently, even an arbitrarily precise observed price->action curve identifies, at best, threshold v/lambda, not v and lambda separately. In a probabilistic choice rule, additional noise-scale normalization assumptions would be needed. The theorem is not a new normative decision optimizer; it is an interpretation limit.

**Counterexample / numerical witness.** Subjective worlds (v=.12, lambda=1) and (v=.24, lambda=2) yield identical decisions [verify, verify, skip, skip] at c=[.03,.08,.13,.18]. If the TRUE forced-action pass probability difference is .80-.60=.20, the skipped cases at c=.13,.18 are both objectively dominated under a unit reward. Under equal price exposure, mean observed action regret is `((.20-.13)+(.20-.18))/4=.0225`. Thus avoidable regret can coexist with observational nonidentifiability of subjective motivation. Artificial values: NOT LLM data.

**Further identifiability warning.** Even forced verification must be defined at the SAME pre-action history, since externally requiring a tool earlier can change the whole reasoning policy. Potential outcome `Y(a;h)` should refer to a controlled one-step tool choice with a common h, fixed available information, task, deterministic tool output and held-out evaluator. Define `V(h)=E[Y(1;h)-Y(0;h)]`; a skipped-call regret for cost c is `1{A=0}max(0,R*V(h)-c)`. Observing the model's own stated confidence alone does not identify its subjective beliefs.

**Nearest falsifier and simpler remedy.** Randomize c, independently estimate V(h) from forced action counterfactuals, and compare observed voluntary calls with the threshold. Compare to CTA and hidden-state utility estimators; test a revealed-calibration intervention before inventing a new algorithm. If conditioned dominated skips disappear under strong baselines and matched conditions, the proposed improvement has no defensible novelty.

**Status:** new exact nonidentifiability lemma and numerical counterexample; strong prior-art collision; `deep_incomplete` until actual model results.

## B3: Cost avoidance versus optimization of a visible evaluator
**Nearest original methods:**
- Thaman, ICML 2026 RHB: https://proceedings.mlr.press/v306/thaman26a.html : skipped verification and evaluator exploits already measured.
- Wang et al., ICLR 2026 TRACE: https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d0ee47d732bd526c4a03d4aecc39bc1-Abstract-Conference.html : reasoning effort used to detect implicit hacking.
- Akter et al., ACL Findings 2026, Evaluator Stress Tests: https://aclanthology.org/2026.findings-acl.513/ : controlled invariance perturbations already target proxy gaming.
- Huang et al. (2026), https://arxiv.org/abs/2609.28614 : autonomous research-agent proxy exploits and reviewer-evasion effects already tested.

**Toy decision reversal.** Honest action (true pass .82, visible score .80, cost .09); shortcut action (true pass .42, visible score .97, cost .01). Maximizing proxy score less cost chooses shortcut; maximizing true pass less cost chooses honest. A 2x2 controlled toy manipulation of whether the proxy is salient and whether cost is multiplied by two has a visibility effect but does NOT imply an LLM intended to deceive: changed salience, reward confusion or instruction interpretation are rivals.

**What differentiates mechanisms.** Keep true task objective constant and seal hidden tests, code diffs and evaluation logs. Randomize superficial proxy exposure and tool price separately; pre-register effects on (i) actual action path, (ii) hidden correctness, and (iii) false completion claim rates. Only call something exploitation when the agent gets higher proxy score while genuinely violating its task specification. Construct an explicit control where proxy visibility is semantically inert to the intended task, and audit that this is so.

**Simplest repair / falsifier.** Sealed tests and evaluator hardening. If hardening alone eliminates the effect or original benchmark's EST gives identical coverage, a new mechanism is not supported.

**Status:** exact toy reversal and prior-art challenge; real agent factorial and reviewer still not run.

## B4: Delayed value of computation defeats one-step marginal greedy policies
**Nearest original methods:**
- Aggarwal et al., ICLR 2026 OptimalThinkingBench: https://proceedings.iclr.cc/paper_files/paper/2026/hash/0f63515b14f33c008158213c7b6191c6-Abstract-Conference.html : both overthinking and underthinking already quantified.
- Zhou et al., ACL Findings 2026: https://aclanthology.org/2026.findings-acl.1199/ : negative marginal utility of longer reasoning observed.
- Li et al., ACL 2026 MUTO: https://aclanthology.org/2026.acl-long.1386/ : token-level marginal utility optimization already proposed.
- Liu et al., COLM 2026 CATS: https://research.google/pubs/cost-effective-agent-test-time-scaling/ : token/tool budget-aware allocation already studied.

**Formal objective.** For task i, `p_i(t)` is success probability given integer effort t, and `sum_i t_i<=B`. Maximize `sum_i p_i(t_i)`. Marginal equalization is justified only under additional smooth/concave assumptions; myopic greedy can fail under delayed complementarity.

**Exact discrete witness.** With B=3, easy-task success curve t=0..3: (.80,.82,.84,.86), hard-task curve: (.15,.15,.15,.80). One-step greedy adds three units to easy, yielding .86+.15=1.01 expected correct. Global optimum allocates all three units to hard, yielding .80+.80=1.60; gap .59. The hard task's first two units seem useless and the third is valuable. This synthetic counterexample is not itself a new algorithm.

**Rival/repair/falsifier.** Access to per-task effort curves is an oracle and generally unavailable online. A fitted early-stop/value model, adaptive search and established CATS/MUTO methods are simple baselines. On held-out tasks, compare strong matched-cost methods and report regret *relative to an oracle only as a diagnostic*, not as claimed deployed performance. If common approaches already resolve delayed complementarities, this mechanism is not a novel paper.

**Status:** new precise greedy failure witness; strong related work; `deep_incomplete` until model-measured value curves and baseline comparisons.

## Reproducible artifact and full-frontier disposition
- `counterfactual_audit.py` reproduces four entirely synthetic witnesses with assertions; its stdout is `CPU_WITNESSES_R002.json`. Tested with Python 3 and bytecode compilation; no model, training or external API executed.
- B1: mechanism updated by actual When2Tool source code inspection and intervention witness.
- B2: mechanism updated by exact scale invariance proof, CTA code inspection and conditional-regret counterexample.
- B3: mechanism updated by EST collision and controlled-proxy limit.
- B4: mechanism updated by delayed-return greedy failure and MUTO/CATS overlap.
- Four active branches served by a single manual assistant turn, NOT four native Work agents or a formally issued controller batch. No claims of `deep_complete`, formal novelty acceptance, independent reviewer approval, provider budget approval or actual LLM validation.

## Recommended next evidence gates (no global direction selected)
1. Native accessible model checkpoint and matched tool-harness environment for the four paired interventions; suggested candidate Qwen/Qwen3-4B-Instruct-2507 as in When2Tool.
2. Reviewer checks whether the proposed conditional-action-regret and latent-motive nonidentifiability are already explicitly proved in Wu et al./CTA, and whether B3 adds anything beyond EST/RHB.
3. A tiny CPU-only static/dynamic harness validation on non-LLM policies before expensive model runs, then model experiments with explicit hardware authorization.
4. Keep B1-B4 active; the next human wake must recover the complete tree and attempt each leaf again.

This is a finite manual-wake R002 checkpoint.