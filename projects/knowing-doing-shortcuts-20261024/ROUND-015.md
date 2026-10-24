# R015: Paired published model outcomes (2026-10-24)

Native tree: revision 3. Active independent peers: B1, B2, B3, B4. Budget not verified; no formal reviewer, GPU or model run.

Source: bennidict23/judged-useless-queried-anyway, experiments/episodes/paper_values.json (Git blob a248f629a30e4ae083a4137a3e01a257d4f03f28). See R015_OFFICIAL_RESULTS_REANALYSIS.json for all eight per-model/per-split paired metrics.

On four models and two 300-question splits, enforced stopping increases the time-matched evidence-response contrast in 8/8 pairs. But task mean6 correctness falls in 3/8 pairs. In Qwen2.5-7B: -9.11pp on test300 and -11pp on fresh300 relative to budget cues. These are published third-party model results; no model was run here.

The authors already studied stopping-vs-success dissociation and their combo arm. The combo budget+enforcement arm dominates the budget-only arm on both metrics in 3/4 test300 models. Combining both strategies therefore is not a new contribution.

B1: Enforced rule removes search/lookup actions; positive stopping contrast does not establish that a free-choice agent internalized evidence. Falsify residual gap using author combo, SafeAct and calibrated stopping.

B2: A USELESS judgment is not the causal value of another tool call versus finishing. Need matched pre-decision world states, actual tool execution effects, and question-level wins/losses.

B3: SpecBench outer AIDE selects candidates from public score; keep a frozen candidate pool and separately intervene on the inner policy to avoid attributing selection artifacts to LLM intent.

B4: Paper's budget arm communicates the fixed eight-action cap; it does NOT randomize real computation. Benchmark against CATS/Conformal Thinking with randomized actual limits.

Local R015 zip contains the actual reanalysis script, paired gz JSONL evaluator, reference CSV and 17 passed CPU tests. Original author binary gz episode rows were not retrieved here. No paired-outcome CI is claimed; the two splits are not independent models.

Next human wake: read this record plus R015_OFFICIAL_RESULTS_REANALYSIS.json; restore B1-B4, inspect binary author episodes in a capable runtime, seek separate independent review. Preserve native tree revision and budgets.