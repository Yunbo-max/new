# R004 research findings (2026-10-24)

Science mode: full-frontier with four active independent branches. We did not run a language model, training, a remote GPU, or an independent reviewer.

B1. A tool necessity label based only on failure without a tool does not establish that a tool would help. A constructed case can fail with both actions.

B2. Observing action under different prices does not identify a model's subjective incentive. Conditional expected improvement needs paired decision-state interventions and repeated trials.

B3. A difference between an easy visible task set and a harder held-out set can arise without policy gaming. Compare conditions with independently matched task difficulty and evaluate actual outcomes.

B4. A requested inference budget must not be confused with consumed tokens or useful computation. Estimate success as a function of assigned budget before modeling stopping policies.

A synthetic validation suite was executed in the model's V8 calculation environment over 192 complete matched triplets (576 records). Seven checks passed, including incorrect-history and duplicate-entry rejection. Its constructed outputs are NOT empirical LLM metrics.

Paper anchors: When2Tool 2605.09252; Calibrate-Then-Act 2602.16699; SpecBench 2605.21384; BAITBENCH 2608.30724; UK AISI on test-time compute.

All four leaves remain eligible. Next: independently validate experimental treatment placement and evaluator isolation, then prepare a native-model runner once resource gates are confirmed.
