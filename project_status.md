# Unlearning Probes — Project State

## Research question

Do unlearning methods achieve forgetting by hiding knowledge behind a recoverable
linear direction (a refusal/membership-like gate)? If so, ablating that direction
should recover forgotten knowledge "for free" — a robustness failure. The refusal
framing has now been tested THREE ways (safety, epistemic, and direct alignment) and
answered in the negative; see Findings 5–7.

## Setup

- Model: Llama-3.2-1B-Instruct, TOFU forget10 checkpoints from open-unlearning
- Base: tofu*...\_full | Oracle: tofu*...\_retain90 (intended honest-ignorance reference)
- Methods: GradDiff, NPO, AltPO, SimNPO (output-preference); RMU (representation-targeting)
- Hardware: M1 Max 32GB, MPS, fp16. transformers 5.9.0 (hook behavior matters — see Lessons).
- Dataset fact (verified): TOFU = 200 fictitious GPT-4 authors x 20 QA each (4000 rows).
  forget10 (400) and retain90 (3600) are a DISJOINT author partition. Oracle was
  fine-tuned on retain90 only, so it genuinely never saw forget10 authors.

## Reference points (forget-answer mean log-prob)

- Base ~ -0.4 to -0.7 (knows it) | Oracle ~ -2.4 (measured this session with the
  sweep's own scorer; status-note value of -2.5 confirmed). USE THE SAME SCORER when
  comparing to oracle — a log-prob is only meaningful relative to its scorer.
- Baselines: GradDiff -0.5, NPO -0.7/-1.0, SimNPO -1.3/-1.4, AltPO -3.1/-3.3, RMU -7.9/-8.3
- GradDiff/NPO/SimNPO barely suppressed -> controls. AltPO/RMU genuinely suppressed -> live cases.

## KEY FINDINGS

### 1. Probe direction is causally inert; difference-in-means is causal (separation != mechanism)

- Ablating the PROBE-NORMAL forget/retain direction: ~0 effect for ALL methods,
  even RMU's perfectly-separating layer-10 direction. Direction verifiably removed
  (proj -> ~0, norm unchanged) yet log-prob doesn't move.
- Ablating the DIFFERENCE-IN-MEANS direction: RMU jumps -7.9 -> -2.2 (oracle level);
  AltPO/SimNPO/NPO stay flat.
- => The probe normal is a CLASSIFIER boundary that separates but does not cause.
  The mean-shift direction is the causally load-bearing one. Core methodological result.

### 2. RMU recovery is COHERENCE restoration, not knowledge recovery (single-point test)

- RMU baseline generation on forget questions = degenerate token-salad (20/20).
- Under diff-in-means ablation at the single c=gap point: fluent, on-topic English,
  but factual recall does NOT return — ~1 correct fact across 20 forget questions,
  the rest confabulate wrong specifics.
- => At c=gap, the -7.9 -> -2.2 log-prob "recovery" was COHERENCE, not knowledge.
  Mean log-prob overstates unlearning reversal; generation reveals the gap.
- Caveat (still open): "facts not recovered by THIS intervention" != "facts destroyed."

### 2b. MAGNITUDE SWEEP + FULL AUDIT — RESOLVED: confabulation, not recovery

- Swept translation magnitude c on the FIXED diff-in-means direction, 0 -> 2\*gap,
  recording per-c mean log-prob (coherence) AND generated fact-hit rate. c=0
  anchor reproduced -7.917 (matches known -7.9).
- FULL curve is an INVERTED-U; coherence and fact_hit peak TOGETHER at c/gap~1.2:
  c/gap logprob fact_hit
  0.00 -7.92 0.05
  0.60 -4.39 0.34
  1.00 -2.66 0.55
  1.20 -2.49 0.57 <- peak (both)
  1.60 -3.10 0.41
  2.00 -4.22 0.17
  Overshoot DEGRADES both -> rules out "matcher just tracks pure coherence with no
  content"; but the coupling is exactly what coherence-ENABLED confabulation predicts
  (you can't fabricate plausible text from token-salad either).
- AUDIT (full forget set, 400 QA at the c/gap=1.2 peak): 228 matcher-hits, but ~0
  state a CORRECT discrete fact. Decomposition: ~50 degenerate (matcher
  false-positives, e.g. "a) Male b) Female" loops grazing "Female"); ~75 matched
  only on SCAFFOLDING tokens (father/works/author); the rest matched on GENRE/STYLE
  schema words ("historical","romance","vivid") the model confabulates for a whole
  CLASS of TOFU authors, not recall. Names wrong (Hsiao Yun-Hwa -> "Wei Wei Li"),
  professions wrong (civil engineer -> "renowned author"), awards fabricated.
- => Finding 2 STANDS and STRENGTHENS. The fact_hit rise is coherence-enabled,
  schema-driven CONFABULATION grazing a token-overlap metric — the SAME house-style
  fabrication the oracle exhibits with no unlearning at all (Finding 6). The sweep's
  real contribution is methodological: token-overlap knowledge metrics are
  UNRELIABLE on TOFU because the benchmark manufactures fluent schema-consistent
  fakes. Confirm knowledge claims with the gold-string log-prob or relearning, not
  generation+matcher.
- Caveat carried forward (now RESOLVED by Finding 8): read tests can't tell
  "unreachable" from "destroyed."
- Artifacts: data/sweep_magnitude_rmu.json, results/sweep_audit_peak.json,
  sweep figures (twin-axis, heatmap, coherence-vs-knowledge scatter).

### 3. RMU norm-inflation mechanism observed (block 1.28x vs MLP 1.83x at layer 10)

- Norm inflation present but MODEST in this checkpoint (c=10 small; paper used 6.5-300).
  Suppression is a MIX of norm-inflation + directional steering. Don't overclaim.

### 4. Taxonomy split

- RMU (representation-targeting): suppression partly reversible via diff-in-means —
  coherence restored (Finding 2/2b). AltPO (output-preference): NOT reversed even by
  the causal direction type that moved RMU. Distinct mechanisms.
- Geometry shift ratio splits methods three ways and BREAKS the naive
  output-pref vs representation-targeting dichotomy: NPO ~1.0 (generic drift),
  GradDiff ~1.15, SimNPO 2-3, RMU 4-5. NPO and SimNPO (same family) behave oppositely.

### 5. SAFETY refusal does not gate unlearning (v1/v2/v3, honest diff-in-means)

- v1 = mean(refused-harmful) - mean(harmless); v2 = all-refused - all-complied;
  v3 = mean(refused-harmful) - mean(complied-harmful) (within-harmful, purest).
  ALL are honest difference-in-means (NOT probe normals) — so the negatives are real,
  not separation artifacts.
- v1 ENCODES HARMFULNESS, not the refusal decision (complied-harmful clusters with
  refused-harmful, far from harmless — replicates "LLMs Encode Harmfulness and Refusal
  Separately" on 1B). v3 is purer but has NO SIGNAL on benign TOFU (forget questions
  sit in the harmless region). Harmless-refused cell is empty (model won't refuse
  benign prompts; instruction-forced refusal = "told to refuse", not genuine).

### 6. EPISTEMIC refusal can't be extracted behaviorally — NEW

- Pivoted to epistemic refusal ("I don't know") as the behaviorally-correct axis for
  benign TOFU. Spot-checked the ORACLE on forget10 authors (genuine ignorance is
  ground truth, since it never trained on them).
- RESULT: 10/10 confident, schema-consistent CONFABULATIONS, zero "I don't know"
  (e.g. invents father's profession inconsistently across questions: dermatologist /
  dietitian / civil engineer). The ignorance cell is EMPTY — same wall as
  harmless-refused, so an epistemic-refusal direction can't be built from
  naturally-occurring ignorance on this setup.
- WHY (defensible, no citation needed): oracle was fine-tuned ONLY on confident TOFU
  biographies (no abstention examples), which suppresses abstention; and calibrated
  uncertainty is weak at 1B scale. The benchmark's homogeneous GPT-4 house style lets
  any retain-trained model fluently fake forget authors. This is the SAME phenomenon
  as Finding 2's coherence-without-facts.
- (Generational "later models trained to abstain" is a HYPOTHESIS, not asserted — would
  need a source on Llama 3.2's abstention tuning.)

### 7. Direct alignment test: refusal direction is NOT special vs unlearning shifts — NEW

- Tested cosine(v3, base->unlearned MEAN-OFFSET) per method at layer 14, benchmarked
  against a RANDOM-GAUSSIAN floor (mean|cos|=0.018, p99=0.054).
- Naive floor flagged 4/5 ABOVE: GradDiff 0.106, AltPO 0.092, SimNPO 0.127, RMU 0.076
  (NPO 0.008 at chance). Tempting but WRONG to read as piggybacking.
- TELLS it's a confound: (i) all 5 cosines POSITIVE (random overlaps are sign-
  symmetric -> a shared component pulls every offset the same way); (ii) the band is
  uniform across method families, incoherent with any mechanistic story (RMU above,
  NPO at chance — opposite of the prediction).
- DECISIVE CONTROL: a refusal-FREE direction (diff-in-means of a random split of the
  HARMLESS prompts) scored cosines against the SAME offsets that BEAT v3 on every
  method (GradDiff 0.186, NPO 0.087, AltPO 0.158, SimNPO 0.202, RMU 0.143).
- => v3 aligns with unlearning shifts NO MORE than (in fact LESS than) an arbitrary
  benign direction. The "above floor" was shared activation-space structure, not
  refusal. The random-Gaussian floor was too permissive because real offsets are not
  isotropic. CLEAN NEGATIVE.
- Artifact: results/alignment_refusal/v3_vs_unlearning_alignment.png (v3 vs control bars).

### 8. RELEARNING-SPEED: instrument unstable; latent-vs-destroyed UNRESOLVED (lr retry pending)

- The only WRITE test (read tests can't distinguish "unreachable" from "destroyed").
  LoRA (r=8, q/v) fine-tune on N authors; score teacher-forced gold-answer log-prob
  vs step. Conditions: rmu_forget (test), base_invented (from-scratch floor),
  base_forget (ceiling). VALIDITY GATE: the ceiling MUST stay flat — if fine-tuning
  degrades facts the model already knows, the instrument can't measure recovery.
- RUN 1, n=5, 50 steps, lr=1e-4: RMU rocketed -8.30 -> -1.27 by step 25 while floor
  stayed flat (-3.02 -> -3.00), ceiling flat (-0.67). ~100x gap-normalized rate.
  Read alone: LATENT. But n=5 = sample of one author-set; not yet robust.
- RUN 2, n=20, 50 steps, lr=1e-4: effect VANISHED. RMU barely moved then drifted DOWN
  (-7.92 -> -8.84). Controls still flat. => n=5 did not replicate at scale.
- RUN 3, n=20, 200 steps, lr=1e-4 (scales steps with data to match per-fact exposure):
  DILUTION RULED OUT. With matched exposure RMU STILL did not relapse — wandered
  -8 to -9.9, ended -8.11, never recovered. AND the CEILING DEGRADED: base_forget
  -0.68 -> -1.56 over 200 steps; base_invented stayed ~flat. => the LoRA setup is
  itself DESTRUCTIVE/UNSTABLE (known facts decay, novel facts don't stick), which
  INVALIDATES the instrument and undermines confidence in the Run-1 n=5 signal (likely
  an unstable upward transient, not genuine re-exposure).
- => CURRENT VERDICT: latent-vs-destroyed is UNRESOLVED, and the relearning test AS
  CONFIGURED cannot settle it — the fine-tuning is too unstable (failed the ceiling
  validity gate) to trust any relearning rate. Do NOT claim latent OR destroyed.
- RUN 4 (RUNNING OVERNIGHT): n=20, 400 steps, lr=2e-5 (5x gentler, open-unlearning's
  training-lr range) + three-tier dense-early checkpoint schedule. HYPOTHESIS: gentler
  lr stops the destructive drift. READ TOMORROW IN THIS ORDER:
  (1) ceiling flat near -0.68 across 400 steps? -> instrument VALID, then read RMU.
  RMU climbs fast vs flat floor -> LATENT; tracks floor -> DESTROYED.
  (2) ceiling still decays -> instability is NOT lr-driven; the test needs a
  structural fix (full FT / regularization / different target modules), and
  Finding 8 closes as "test inconclusive, documented, fix path identified."
- LESSON (now two-pronged through-line): separation/overlap isn't mechanism until you
  control for default structure (Findings 1/2b/7); AND a striking result isn't a result
  until it survives scale-up AND its measurement instrument passes its own validity
  gate (Finding 8). The ceiling control catching the instability is the experiment
  working correctly.
- Artifacts (config-aware names): results/relearning/auth{N}.step{S}.lr{LR}.\*.json + logs.
  Run 1 n=5 JSON was lost to filename collision (since fixed); n=5 recoverable from log.

## HEADLINE (for the talk)

Solid, control-backed results:

1. REFUSAL IS NOT THE GATE (answers Sam Dower's question). Four converging negatives:
   geometry — no coherent per-question gate beyond a constant offset (Finding 1);
   safety refusal encodes harmfulness, silent on benign TOFU (5); epistemic refusal
   can't be extracted, model confabulates rather than abstaining (6); the purest
   refusal direction aligns with unlearning shifts NO MORE than an arbitrary
   direction, once a refusal-free control replaces the too-permissive Gaussian
   floor (7).

2. RMU's APPARENT activation-space recovery is a COHERENCE ARTIFACT, not knowledge.
   Diff-in-means ablation/translation restores fluent on-topic generation but ~0
   correct facts (full-set audit, 400 QA); the rising fact_hit is schema-driven
   CONFABULATION grazing a token-overlap metric — the same house-style fabrication
   the oracle exhibits with no unlearning at all (Findings 2/2b/6).

In-progress / honest open result:

3. LATENT-vs-DESTROYED is UNRESOLVED (Finding 8), and the relearning instrument is
   currently UNSTABLE. A striking ~100x-faster relapse at n=5 (-> latent) did not
   replicate at n=20; scaling steps to match per-fact exposure (200 steps) ruled out
   the dilution explanation AND revealed the LoRA setup is destructive — the CEILING
   control (facts the model already knows) DEGRADES under fine-tuning, which invalidates
   the test. A gentler-lr run (2e-5, 400 steps) is testing whether stability returns;
   the ceiling-stays-flat validity gate decides whether the test can answer at all.
   Frame as: "the write-test looked decisive, but failed its own validity check — the
   measurement apparatus degrades known facts — so latent-vs-destroyed stays open
   pending a stable training config." The ceiling control catching this is the
   experiment working correctly, not a setback.
   headline and it didn't hold at scale" — NOT as a settled latent claim.

METHODOLOGICAL THROUGH-LINE (the real contribution): a separating OR overlapping
measurement is not evidence of mechanism until you control for the structure that
produces separation/overlap by default — AND a striking result is not a result until
it survives a scale-up AND its instrument passes its own validity check. Manifestations:
probe-normal separates but isn't causal (1); the Gaussian floor flagged everything until
a refusal-free control revealed shared structure (7); mean log-prob "recovery" conflates
coherence with knowledge (2/2b); and the relearning ceiling-control caught that the
fine-tuning instrument itself degrades known facts (8). Same lesson each time: the
control is what tells you whether the measurement means anything.

## Methodological lessons (do not re-learn)

- transformers 5.x IGNORES return-based forward-hook edits -> hook must mutate IN-PLACE.
- Intervene at ALL sequence positions. Teacher-forced MEAN log-prob, off-by-one.
- Each model needs its OWN direction from its OWN sweep.
- Match intervention site to the question (block output vs MLP).
- PROBE NORMAL != DIFFERENCE-IN-MEANS != causal direction. Test diff-in-means before
  concluding "no recoverable direction."
- Log-prob recovery can be coherence, not knowledge — confirm with GENERATION.
- Fact-hit token-overlap matchers OVER-count: exclude prompt tokens, and a generous
  one-token threshold makes a FLAT curve conservative but a RISING curve weak evidence.
  Audit by eye before quoting a knowledge-recovery number.
- A cosine/projection ~0 is over-determined in high-dim; benchmark against a CONTROL
  direction (refusal-free), not just a random-Gaussian floor. All-same-sign cosines
  across items = shared-component confound.
- Eliciting BEHAVIOR (does it say "I don't know"?) needs the CHAT TEMPLATE; teacher-
  forced scoring deliberately does not (baseline-consistency vs naturalness).
- MEMORY (M1 32GB): only ONE 1B fp16 model fits comfortably. A notebook free_model
  helper that receives the OBJECT cannot free a notebook GLOBAL — del the global NAME
  (or pass names/ns), THEN gc.collect()+empty_cache(). `del local` must precede the
  reclaim, or two models briefly co-reside. Restarting a kernel frees that kernel only
  (separate processes) — but don't run two model jobs concurrently; they share the GPU.

## Codebase (src/)

- model_loader.py, hooks.py, probes.py, layer_sweep.py, intervention.py,
  recovery_experiment.py, geometry.py, dashboard.py — stable (see prior notes).
- config.py — NEW. Single source of truth: CHECKPOINTS, RETAIN, BASE_MODEL,
  per-model intervention layers (RMU=10, others=14), checkpoint()/layer_name()
  accessors (raise on typo), LIVE_CASES/CONTROLS. Inert on import.
- magnitude_sweep.py — NEW. run_sweep (dual logprob+fact-hit over c), default_fact_hit
  (prompt-token-excluded, generous screening metric), plot_sweep / heatmap / scatter.
- refusal_alignment.py — NEW. build_v3 (from labeled CSV), build_control_direction
  (random harmless split, refusal-free), run_alignment (v3 + control vs offsets,
  one model resident at a time), alignment_table (control comparison + sign check),
  plot_alignment (v3-vs-control bars). Saves to results/alignment_refusal/.
- invented_authors.py — NEW. Deterministic schema-matched, exotic-name (low
  pretraining-mass) author QA, the from-scratch control set for relearning.
- relearning.py — NEW. LoRA relearning loop, per-checkpoint saving (crash-safe),
  three matched conditions, plot_relearning (curves + per-author spread),
  relearning_rates (gap-normalized early-slope verdict: latent/destroyed/partial).
- audit_peak.py — NEW. Reruns translation at given c/gap on the full forget set,
  prints generations vs gold with hit label + degenerate (false-positive) flag.
  Parameterized CLI (--c list, --model, --layer, --n).

## Notebooks

- 01 sanity, 02 magnitude sweep (NEW), 03 refusal behavior check (AdvBench/ALPACA,
  dual Ollama judges), 04 refusal direction (v1/v2/v3), 05 geometry scan,
  06 epistemic refusal (NEW — oracle ignorance gate check, reuses 03's judge pipeline).

## Open / next steps

SOLID & CLOSED: refusal-is-not-the-gate (Findings 1/5/6/7), coherence-artifact audit
(2/2b). Both control-backed, done.

IMMEDIATE (relearning thread, to close it one way or the other):

1. READ RUN 4 (n=20, 400 steps, lr=2e-5, running overnight). Validity gate FIRST:
   did base_forget ceiling stay flat near -0.68? If yes -> instrument valid, read
   RMU's gap-normalized rate for the latent-vs-destroyed answer. If no -> instability
   is structural, not lr; close Finding 8 as "test inconclusive, fix path identified."
2. IF instrument valid but want robustness: AltPO negative control (does the method
   that RESISTED diff-in-means also relapse, or is RMU special?). Same loop, --model AltPO.

NEW DIRECTION (Sandy's post-talk steer toward GEOMETRY; ties to the two project PDFs,
Shape of Beliefs / Linear Field Probing, GoodFire): 3. The whole project keeps hitting one wall: difference-in-means is a LINEAR, GLOBAL
operation (base centroid -> unlearned centroid, a straight chord). Findings 1/2b/7
are all "this linear direction separates/moves things but isn't the mechanism."
Shape of Beliefs' thesis is exactly this: beliefs live on CURVED manifolds, global-
linear directions are inadequate, linear steering moves reps OFF-manifold. Hypothesis:
unlearning is a CURVED trajectory, and diff-in-means fails because it approximates an
arc with a chord.

- EXTENSION A (start here — needs NO new infra): manifold curvature on the EXISTING
  checkpoints via Linear Field Probing. Fit LOCAL linear directions in patches across
  the forget cloud; if the required direction VARIES across the cloud (curvature),
  that quantitatively explains why global diff-in-means ablation was inert/coherence-
  only. First step: read the LFP method from the Shape of Beliefs PDF (don't
  reconstruct from memory) before implementing.
- EXTENSION B (bigger; needs intermediate checkpoints): reconstruct the centroid's
  PATH through representation space across unlearning epochs and measure its curvature.
  open-unlearning publishes only FINAL checkpoints, so intermediates must be
  REGENERATED via their training script (lr 1e-5, 10 epochs, documented configs;
  wants a real GPU). Prior art: "Revisiting the Past: Data Unlearning with Model State
  History" does the weight-space cousin (extract theta_1 - theta_0 from an intermediate
  and reapply) — found it usually doesn't improve performance, so study the GEOMETRY,
  not as a performance win. TOFU's own paper already plots unlearning TRAJECTORIES in
  eval-metric space; doing it in REPRESENTATION space is the novel contribution.

DEFERRED / CAVEATS: 4. Phase 2 (situational awareness): clean SA direction blocked by the same behavioral-
elicitation wall at 1B (model confabulates / won't abstain). SA is itself a geometry
question, so it may re-enter via Extension A/B on a larger model. 5. CROSS-CUTTING write-up caveat: all results are 1B + TOFU. TOFU's homogeneous GPT-4
house style drives the confabulation (2b/6); scale and benchmark-diversity are the
obvious generalization questions.
