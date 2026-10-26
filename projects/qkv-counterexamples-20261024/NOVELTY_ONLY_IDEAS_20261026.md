# 2026-10-26 · Novelty-first QKV Research Reset (IDEAS ONLY)

Scope: original user-selected QKV architecture research; repository Yunbo-max/new. This is an **ideation and literature screen**, NOT an R016/R017 science checkpoint, not a new reviewer verdict, and not an experiment. It leaves CURRENT.json and all six active/seven parked branches unchanged.

## Verdict on R001–R016
No defensibly first-in-the-world novel QKV architecture has been established. Mathematical counterexamples include MomentKV v1's unqualified operator guarantee and suspected Appendix Eq35 algebra issue, anti-correlated head errors and BF16 radial collisions. Existing work already treats deterministic post-deletion nonidentifiability (Xie 2026), output-aware cache damage (Behavior-Preserving KV, OptR), cross-layer cancellation (WitCert), query-future forecasting (LookaheadKV, AgentKV), and compression-friendly gauge transforms (Codec-Gauge). These are neither unique architectures nor end-to-end model results. The target NOW is conceptual novelty, not more toy experiments.

## Highest-upside architecture candidate: Transactional / Causally Editable QKV

### Fundamental gap
An ordinary causal decoder *appends* immutable per-token K_i,V_i. When an earlier source is updated/retracted, merely appending a replacement leaves old information encoded in many downstream cached K/V. Re-prefilling/repairing the downstream window is possible but changes many cached states. Can a new attention architecture make **versioned semantic revisions a first-class operation** without reliving all prior tokens?

### Candidate operator (conditional design, NOT existing Qwen parity)
Separate source-independent immutable per-token codes (k_i^0,v_i^0) from a mutable source-state dictionary z_g. At read time:
  k_i(z)=k_i^0 + Σ_{g in deps(i)} a_{ig} U_K z_g
  v_i(z)=v_i^0 + Σ_{g in deps(i)} b_{ig} U_V z_g.
An edit, withdrawal or revision to source g changes only z_g; all relevant keys/values are late-bound when queried, while the immutable token cache is reused. Sparse a_ig,b_ig and shared low-rank U limit side metadata. This is EXACT for the **defined factorized read operator**, not an exact edit of an arbitrary pretrained Transformer: deep contextual downstream states and generated tokens may carry stale causal information unless the architecture is designed to keep mutable dependencies out of the immutable stream.

Precise architectural research question: Is there a nontrivial multi-layer factorization with bounded per-token dependency metadata that permits exact or bounded-error semantic source revision across long causal continuations while preserving ordinary QKV retrieval? Contrast with a standard attention cache requiring downstream replay.

Why counterintuitive: important old K/V need not be recomputed or overwritten after a correction; instead their **meaning at read time** changes while the stored base remains unchanged.
Distinctness target: unlike EditLocal / Contiguity, Not Importance (arXiv:2609.17983), not **which old rows to recompute**; unlike DynaRule (EACL Findings 2026), not simply injecting/retrieving latest rules but an explicit input-edit-to-QKV state-transition algebra. Novelty is *unverified*—late-binding KV and modular memory are adjacent prior art.
Key conceptual falsifier: any existing architecture with the same factorized late-binding state and edit semantics, or a no-go showing nontrivial causal dependency cannot be kept low-rank except by recomputation.

## Alternative 2: Dual-space Witness KV (query-space, not token-space, certificate)
Let full head output F(q)=N(q)/Z(q), approximate compressed read Fhat(q). During pre-deletion time, choose distinguished query witnesses q_j and store exact F(q_j), together with valid original key/value norm envelopes allowing Lipschitz bound L_F; Fhat has its own L_Fhat. For a new query q and nearest witness j:
  ||F(q)-Fhat(q)|| <= ||F(q_j)-Fhat(q_j)|| +(L_F+L_Fhat)||q-q_j||.
This is a strictly conditional certificate **without rereading deleted K/V**. The new cache stores *where it knows it is right* in reachable-query geometry, rather than more global KV moments. Optimize witnesses as a cover of reachable queries or a counterexample-seeking set. Ordinary epsilon nets and Lipschitz interpolation are textbook math, and a curse of intrinsic query dimension may make certificates vacuous. Source collisions include Xie (2607.21475), Certified Sparse Attention (2610.24991, scans original keys), Jacap (2609.08131), predictive equivalence theory and future-query eviction. No unique method established. Falsifier: certificate coverage negligible, known query landmarks/probe-cache method matches it, or envelope bytes exceed retained exact KV cost.

## Alternative 3: Antithetic Head-Code Quantization
Ordinary cache quantizers try to minimize each head's error; instead deliberately choose **correlated quantization errors** so the *sum after output projection* cancels:
  minimize over coupled code choices C_h: E_q ||Σ_h W_O,h ε_h(q;C_h)||²  subject to Σ_h bits(C_h)<=B.
One head may be quantized WORSE individually to reduce full-block output error. A common random seed and coupled stochastic-rounding codebooks can create negative cross-head covariance without storing extra per-token side channels in principle. Counterintuitive novelty boundary: a *constructive shared-randomness error-correlation operator* with exact byte parity, not just optimizing post-WO MSE or noticing cancellation.
Very close works: WitCert (2607.28699) observes cross-layer cancellation, OptR (2608.02691) and AATC (2608.14191) are output-aware KV quantization, KV-Surgeon (2610.25933) jointly compensates removal. An explicit similar coupled-dither scheme would kill the novelty. No demonstrated gains or completed exhaustive search.

## Retire these as novelty claims
- Future-query-based eviction by itself: Expected Attention (2510.00636), LookaheadKV (ICLR2026), AgentKV (2609.14872).
- Rotation/gauge KV compression by itself: Codec-Gauge (2607.20538), RACE (2610.28385), KVTC.
- Output-aware compaction and safe-gating by itself: OptR, Behavior-Preserving KV (2610.06479), Fast KV Compaction (ICML2026), KV-Surgeon, Xie, WitCert.
- Source-copy deduplication alone: Copies or Sources (2610.06192), Epistemic Sybil Resistance (2609.01873), PQ-EID (Future Internet 2026).

## Selection, not invented certainty
Research preference by **conceptual mechanism distance from the crowded cache-compression frontier**: 1) Transactional QKV, 2) Query-Witness KV, 3) Antithetic Head Coding. These are original proposals assembled in this audit, not verified claims that “nobody has done them.” A finite public-literature search cannot certify global novelty. No experiments are proposed as prerequisites for this ideas-only request. We should not increase round/served counters, claim an accepted architecture, run GPUs, or dispatch a reviewer as part of this audit.

## Source links for closest mechanisms
- EditLocal: https://arxiv.org/abs/2609.17983
- DynaRule (2026 Findings EACL): https://aclanthology.org/volumes/2026.findings-eacl/
- WitCert: https://arxiv.org/abs/2607.28699
- OptR: https://arxiv.org/abs/2608.02691
- Xie eviction certificates: https://arxiv.org/abs/2607.21475
- Certified Sparse Attention: https://arxiv.org/abs/2610.24991
- Jacap: https://arxiv.org/abs/2609.08131
- Codec-Gauge: https://arxiv.org/abs/2607.20538
- AgentKV: https://arxiv.org/abs/2609.14872
- Behavior-Preserving: https://arxiv.org/abs/2610.06479
- KV-Surgeon: https://arxiv.org/abs/2610.25933
- Copies or Sources: https://arxiv.org/abs/2610.06192
- Epistemic Sybil Resistance: https://arxiv.org/abs/2609.01873
