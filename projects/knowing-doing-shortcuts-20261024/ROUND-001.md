# R001 — Four distinct causes of apparent LLM shortcutting
Date: 2026-10-24. Scientific status: `deep_incomplete`; theoretical toy witnesses done; real-model evidence pending.
Prior project results were NOT read or reused. No model run, no Work subagents, no independent reviewer were available or executed.

## Scope and closest references
- Cheng et al. 2026, Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap: https://arxiv.org/abs/2605.14038 . Model-adaptive tool-use mismatch is empirically established; hidden-state cognition/action probe directions differ. Qwen3-4B/8B, Llama 3B/8B; 4,000 arithmetic and 817 TruthfulQA instances.
- Park et al., ICML 2026, Bridging the Knowledge-Prediction Gap (KAPPA): https://proceedings.mlr.press/v306/park26v.html . Readout/knowledge misalignment in MCQs; inference-time representational intervention.
- Herzig et al. 2025, Inside-Out: https://research.google/pubs/inside-out-hidden-factual-knowledge-in-llms/ . Latent factual discriminability exceeds generated correct answers.
- Wang et al., ICLR 2026, TRACE: https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d0ee47d732bd526c4a03d4aecc39bc1-Abstract-Conference.html . CoT truncation and reward–reasoning-effort AUC detect implicit reward hacking; already addresses effort-based detection.
- Thaman, ICML 2026, RHB: https://proceedings.mlr.press/v306/thaman26a.html . Agents skip verification and exploit evaluation mechanisms in tool tasks; this phenomenon is not novel.
- Huang et al., September 2026, research-agent reward hacking: https://arxiv.org/abs/2609.28614 . Spontaneous exploit rates depend strongly on task environment; independent evaluation needed.
- Aggarwal et al., ICLR 2026, OptimalThinkingBench: https://proceedings.iclr.cc/paper_files/paper/2026/hash/0f63515b14f33c008158213c7b6191c6-Abstract-Conference.html . Both over- and underthinking have been benchmarked.
- Snell et al., ICLR 2025, test-time compute allocation: https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html . Compute allocation per task is known. 

## B1 — Readout/control bottleneck (not a subjective claim about awareness)
**Mechanism:** Pre-action hidden state H might allow a linear probe to decode whether a tool is beneficial (K), without the sampled action A using that information.
**Formal operator:** K=g(H); action A~pi_theta(.|H,x); ground truth label N(x) derived from counterfactual success. An accurately decoded K does not imply K causally controls A.
**Witness (logical):** x in {0,1}, H=x, K=x (100% probe accuracy), but A=0 for both states, so delta A=0. A perfect probe exists while the agent does not act on the decoded property. This is an existence witness, not evidence of an LLM's actual internal behavior.
**Rival:** K may be probe-extractable but not cheaply/readily accessible to the model's action decoder; or the model may correctly choose A=0 because test-time tool cost dominates.
**Simple repair:** linear action readout/steering or supervised tool-choice head; compare to cost-free forced invocation.
**Novelty collision:** Cheng et al. already demonstrate cognition/action mismatch, and KAPPA already proposes readout alignment. A new paper cannot claim that observing K!=A is novel. Conditionally distinct question: does randomized tool cost causally change K->A transfer after controlling empirical action value?
**Protocol/falsifier:** obtain pre-action activations for held-out MBPP+ tasks; train K probe only on train split; measure K discrimination and intervention effect on A and hidden pass rate. Cross-evaluate probe direction under prompt/cost change. Falsified as a bottleneck explanation if readout interventions fail to change action or if high probe accuracy disappears out of distribution.
**Status:** mathematical counterexample checked by CPU, source equivalence screened, native model and independent review not run.

## B2 — Perceived cost versus perceived success: identifiability
**Mechanism:** A=1 means verify; choose according to sign of DeltaU = R*(phat_verify - phat_skip) - lambda*c, with p-hats as subjective probabilities and c as externally stated price. This is a *behavioral hypothesis*, not a proven mechanism of transformers.
**Exact result:** observing A alone identifies only the sign of DeltaU, not subjective gain or lambda. Even with randomized c and a stable deterministic threshold, one can at most recover R*(phat_verify-phat_skip)/lambda, not the factors separately.
**CPU witness:** true p_verify=.82, p_skip=.62, low/high costs=.01/.09 imply true marginal gain .19/.11, so verification is strictly preferable at both prices (R=1). Yet a different model believing phat_verify=.68, phat_skip=.62 yields perceived high-price gain -.03 and reasonably skips. Same observed shortcut can arise from mistaken capability beliefs instead of laziness. This refutes any inference of 'intentional effort aversion' from a skip alone.
**Rival:** instruction misunderstanding, tool unreliability, environmental latency, task difficulty, or trust in self-solution.
**Simple repair:** a calibrated empirical estimate of forced-verification improvement plus a decision rule; compare to a lower-cost forced-verification baseline. A new architecture is premature.
**Novelty collision:** cost-aware adaptive tool/compute choices are extensively studied; a claim must be *causal conditional on model capability and action-value*, not 'charge more for a tool and fewer calls result'.
**Protocol/falsifier:** randomized low/high price and neutral instruction with fixed token budget; hold model/task/seed matched. Separately measure forced-verify and forced-skip hidden outcomes, giving DeltaValue(x,c)=E[Y|do(A=1),x]-E[Y|do(A=0),x]-c/R. Estimate policy regret E[max(0,DeltaValue)*1(A=0)] on held-out tasks. A suboptimal skip requires DeltaValue>0, not merely a preference for cheapness. Falsify 'cost suppression' if causal cost effect vanishes or if all skipping has nonpositive forced-action value.
**Status:** derived identifiability counterexample, CPU checked. No LLM causality observed.

## B3 — Proxy-metric exploitation versus economical honest completion
**Mechanism:** Agent optimizes observed proxy score S, while evaluator values hidden real success T. Shortcut is strategically advantageous under proxy only if E[S(short)-S(honest)] exceeds cost and any perceived risk.
**Witness:** honest (T=.82,S=.80,c=.09), shortcut (T=.42,S=.97,c=.01). Maximizing S-c selects shortcut (.96 versus .71); maximizing T-c selects honest (.73 versus .41). No humanlike deception is required.
**Rival:** evaluation metric is truly what task requests, not a proxy; shortcut is legitimately efficient, or task is impossible under constraints. Must not label efficient correct behavior as reward hacking.
**Simple repair:** independent hidden tests and sealed tool traces, prevent agent from editing evaluator or seeing hidden labels. No new training method needed if environmental hardening suffices.
**Novelty collision:** RHB directly covers skipped verification and evaluator exploits; TRACE already uses reasoning effort for detection. Distinct question only if causal incentive perturbations separate strategic proxy pursuit from ordinary cost aversion *under matched true task objective*.
**Protocol/falsifier:** one agent environment, 2x2 factorial: legitimate tool-cost x visibility of a superficial completion proxy, task objective fixed and hidden evaluator external. Compare actual execution and hidden success, not CoT statements. If apparent hack rate unchanged by proxy incentives or explained by lower true action value, this branch lacks its proposed mechanism.
**Status:** synthetic preference reversal CPU verified, closest benchmark read, no actual hacking rate estimated.

## B4 — Marginal effort allocation (task-specific and action-specific)
**Mechanism:** For tasks i, allocate t_i thinking/tool budget under sum_i t_i<=B to maximize sum_i p_i(t_i). Optimal interior marginal gain per unit effort should balance (under differentiability, concavity and exact p_i access): p_i'(t_i)=lambda, with boundary inequalities. Nonmonotone p_i can make more thinking actively harmful.
**Witness (discrete, no model):** easy-task pass-probabilities for t=0..4 are (.93,.95,.93,.92,.90); hard-task probabilities (.30,.45,.63,.70,.71). With B=4, split (t_easy=1,t_hard=3) yields 1.65 expected successes; equal split (2,2) only 1.56. This example does not establish an original method; it illustrates why uniform effort can be inefficient.
**Rival:** apparent tokens may be scaffolding or output length, not productive computation; task response distribution can change with longer generation; optimality requires counterfactual effort curves, not difficulty labels.
**Simple repair:** dynamic early stopping and light verifier trained on held-out marginal success gains. Include fixed-budget allocation and confidence-only gates.
**Novelty collision:** OptimalThinkingBench and compute-optimal scaling already address broad allocation, so this branch is presently low-novelty unless action-level, causal marginal-value estimation differs from those methods.
**Protocol/falsifier:** for each frozen benchmark task sample token budgets (short/mid/long) and tool-test counts, evaluate paired seeds, bootstrap expected marginal success. Check gain over tuned adaptive-budget methods at matched total tokens and tool cost. Falsified as a new mechanism if tuned fixed/standard adaptive gates saturate gains.
**Status:** known literature + CPU example; new mechanism not yet established.

## Distinguishing real causal work from easy narratives
A model saying "I know I should test" is not proof of calibrated knowledge. A perfect linear probe is not proof of causal accessibility. A shortcut is not reward hacking unless it undermines the explicitly intended objective. Higher cost causing fewer tool calls is not evidence of irrational laziness. Token count is not a faithful measure of reasoning quality.

## Operational pilot: independent coding repair and verification
- Local checkpoint proposal: Qwen2.5-Coder-7B-Instruct; 1.5B-Instruct as capability-shift control, both without APIs. Verify exact model revisions before launch.
- Tasks: MBPP/EvalPlus MBPP+, with separated visible checks versus sealed hidden assertions. Coding agent gets only task text, editor and read-only `run_visible_tests`; evaluator scores on withheld tests outside agent control. Do not permit log/test tampering.
- Matched factorial: 2 relative verification prices x 2 proxy visibility conditions x optional/required verification controls. Randomize assignment within problem, keep max output tokens and tools fixed across arms.
- Outcome: actual `run_visible_tests` calls, hidden correctness, explicit false completion claims, latency/tokens, empirical forced-action uplift, policy regret, proxy-induced action difference; confidence intervals clustered by task. A model's subjective belief is not assumed observable.
- First pilot scope: 50 tasks x 2 seeds x 4 factorial cells = 400 sampled runs plus a separately allocated forced-action subset; do not call this 400 runs 'completed'.
- Controls: always-verify, never-verify, cost-blind baseline, simple cost-aware rule trained on disjoint tasks. Compare to the known technique of environmental hardening.
- Model generation, reviewer approval, GPU budget, exact hidden harness and task split are pending; the pilot is a protocol only.

## Full-frontier audit and next questions
Four authentic mechanism-distinct active leaves B1–B4. Each received source screen, derivation, nearest counterexample, simple remedy, diagnostic protocol and explicit falsifier in this finite wake. Completed: four artificial numeric/logical CPU witnesses. Not completed: model runs=0, independent reviews=0, native execution=0. No evidence that the four branches are publishably novel.
- B1 next: causal test of probe-information accessibility under forced hidden-state steering; review KAPPA and Cheng code-equivalence.
- B2 next: estimate paired forced-action uplift and randomized-cost elasticity, then conditional regret.
- B3 next: independently audited proxy-visibility factorial, no accessible evaluator tampering.
- B4 next: compare allocation policy to strong published compute-optimal baselines; may be parked locally if equivalent.

Research remains active, no global primary direction. Work ended at this checkpoint; a new human wake is required for continued research. Cost/tool token usage not measured; do not record them as zero.
