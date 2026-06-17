# Unlearning Probes — Project State

## Research question

Do unlearning methods achieve forgetting by hiding knowledge behind a recoverable
linear direction (a refusal/membership-like gate)? If so, ablating that direction
should recover forgotten knowledge "for free" — a robustness failure. Phase 2 extends
the same test to situational-awareness directions (refs: _The Shape of Beliefs_,
_Language Models Recognize Dropout_).

## Setup

- Model: Llama-3.2-1B-Instruct, TOFU forget10 checkpoints from open-unlearning
- Base: tofu*...\_full | Oracle: tofu*...\_retain90 (honest-ignorance reference)
- Methods: GradDiff, NPO, AltPO, SimNPO (output-preference); RMU (representation-targeting)
- Hardware: M1 Max 32GB, MPS, fp16. transformers 5.9.0 (hook behavior matters — see Lessons).

## Reference points (forget-answer mean log-prob)

- Base ~ -0.4 to -0.7 (knows it) | Oracle ~ -2.5 (honest ignorance, recovery target)
- Baselines: GradDiff -0.5, NPO -0.7/-1.0, SimNPO -1.3/-1.4, AltPO -3.1/-3.3, RMU -7.9/-8.3
- GradDiff/NPO/SimNPO barely suppressed -> controls. AltPO/RMU genuinely suppressed -> live cases.

## KEY FINDINGS (validated)

### 1. Probe direction is causally inert; difference-in-means is causal (separation != mechanism)

- Ablating the PROBE-NORMAL forget/retain direction: ~0 effect for ALL methods,
  even RMU's perfectly-separating layer-10 direction (acc 1.00, zero overlap).
  Direction verifiably removed (proj -> ~0, norm unchanged) yet log-prob doesn't move.
- Ablating the DIFFERENCE-IN-MEANS direction: RMU jumps -7.9 -> -2.2 (to oracle level);
  AltPO/SimNPO/NPO stay flat.
- => The probe normal is a CLASSIFIER boundary that separates but does not cause.
  The mean-shift direction is the causally load-bearing one. This is the core
  methodological result: probe separability is NOT causal evidence.

### 2. RMU recovery is COHERENCE restoration, not knowledge recovery

- RMU baseline generation on forget questions = degenerate token-salad (20/20 gibberish).
- Under diff-in-means ablation: fluent grammatical English, on-topic (20/20 coherent).
- BUT factual recall does NOT return: across 20 forget questions, ~1 correct discrete
  fact (Q17 "writes in English"), ~1 partial (Q11 mother's unemployment); the rest
  confabulate wrong specifics (father = "author" not civil engineer; gender = "male"
  not LGBTQ+; fabricated book titles). Subject name only reproduced when already in prompt.
- => Ablation restores GENERATIVE COHERENCE but not the forgotten FACTS.
  The -7.9 -> -2.2 log-prob "recovery" was a COHERENCE artifact, not knowledge return.
- METHODOLOGICAL CONTRIBUTION: mean log-prob overstates unlearning reversal; it
  conflates coherence restoration with knowledge restoration. Generation reveals the gap.
- Knowledge-status caveat: "facts not recovered by THIS intervention" != "facts destroyed."
  Single direction, single layer; latent knowledge could exist but be unreachable this way.
  Definitive test = relearning-speed (latent -> relearns instantly; gone -> from-scratch).

### 3. RMU norm-inflation mechanism observed directly

- RMU steers forget MLP outputs toward a scaled random vector (c\*u, c=10) — published
  mechanism (WMDP, Li et al. 2024; effect partly norm-scaling per their own ablation).
- Measured layer-10 activation norm, forget vs retain:
  - block output (model.layers.10): 7.55 vs 5.88 = 1.28x
  - MLP output (model.layers.10.mlp): 7.35 vs 4.01 = 1.83x (block output dilutes it)
- => Norm inflation present but MODEST in this checkpoint (c=10 is small; paper used 6.5-300).
  Suppression is a MIX of norm-inflation + directional steering, not pure magnitude.
  Don't overclaim "RMU works by norm inflation" — 1.83x supports "present but partial."

### 4. Taxonomy split (the contribution)

- RMU (representation-targeting): suppression partly reversible via diff-in-means —
  coherence fully restored, facts not. Mechanism = norm-inflation + steering.
- AltPO (output-preference): NOT reversed even by the causal direction type that
  moved RMU (-3.28 -> -3.26). Different, more entangled mechanism.
- Different method families suppress through different, mechanistically distinct routes.

## Methodological lessons (do not re-learn)

- transformers 5.x IGNORES return-based forward-hook edits -> hook must mutate IN-PLACE
  (hidden.sub\_(...)). Random-direction ablation sanity check catches a dead hook.
- Intervene at ALL sequence positions (causal attention). Teacher-forced MEAN log-prob,
  off-by-one (logits i-1 predict token i), call model(input_ids=..., attention_mask=...).
- Each model needs its OWN direction from its OWN sweep.
- Hook EXECUTION order = registration order. To capture a post-ablation value, register
  the capture hook AFTER (or inside) the intervention. A submodule that runs BEFORE the
  hooked module won't see the edit (why the MLP-norm-under-block-ablation check was a no-op).
- Match intervention site to the question: ablate at block output to affect OUTPUT;
  ablate at the MLP to affect the MLP's own output.
- PROBE NORMAL (classifier) != DIFFERENCE-IN-MEANS (displacement) != causal direction.
  Always test diff-in-means before concluding "no recoverable direction."
- Log-prob recovery can be coherence, not knowledge — confirm with GENERATION.

## Codebase (src/)

- model_loader.py, hooks.py, probes.py — stable. (probes.py: fixed save_probe signature.)
- layer_sweep.py — optimized to ~2 forward passes via extract_activations_multi (was 32).
- intervention.py — VALIDATED. In-place ablation/translation hooks (all positions),
  teacher-forced log-prob scorer, run_recovery, load_direction, gap_from_metadata,
  diff_in_means_direction (computes mean-shift dir, optional raw gap).
- recovery_experiment.py — run_one_model / run_all_models (saves JSON), summarize_table,
  plot_recovery, plot_means_vs_oracle.

## Artifacts for the talk

- recovery_all_models.json — probe-direction results (all inert).
- diff_in_means.json — diff-in-means results (RMU recovers coherence, others flat).
- 20-example RMU generation transcript (coherence-yes, facts-no).
- Charts to make/insert: baseline-vs-ablated bars w/ oracle line; probe-vs-diffmeans
  comparison; norm-inflation (block 1.28x vs MLP 1.83x).

## Open / next steps

1. Relearning-speed test on ablated RMU — the definitive "facts latent vs destroyed" check.
2. AltPO/SimNPO diff-in-means + per-layer projection control (RMU has it, they don't).
3. Quantify the 20-example factual-hit rate rigorously (define hit, score all forget set).
4. Multi-layer ablation; ablate-at-MLP to confirm norm deflation at source.
5. Phase 2: situational-awareness directions, same recovery test.

## Talk framing (BlueDot lightning + breakout)

Headline: probe separability is NOT causal evidence — the diff-in-means direction is
the causal one. Ablating it restores RMU's COHERENCE but not its KNOWLEDGE; mean
log-prob "recovery" was a coherence artifact. AltPO resists even the causal direction
-> a real method-family taxonomy. Norm-inflation mechanism observed (1.83x at MLP).
Outline file (bluedot_lightning_talk.md) still reflects an earlier framing — UPDATE
before slides.
