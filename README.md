# Unlearning Probes

Mechanistic investigation into _how_ LLM unlearning methods achieve forgetting on
the TOFU benchmark — whether they piggyback on safety-refusal mechanisms, whether
the "knowledge recovery" seen under activation interventions is real recall or
fluent confabulation, and whether forgotten knowledge is destroyed or merely
suppressed (latent).

Model: `Llama-3.2-1B-Instruct` (TOFU-finetuned). Unlearned checkpoints from
`open-unlearning`. All experiments run on a single Apple-Silicon machine (MPS).

---

## What the project found

Three substantive results, each built on an explicit control — the methodological
spine of the project is that **a separating or recovering measurement is not
evidence of a mechanism until you control for the structure that produces it by
default**.

1. **Refusal is not the gating mechanism.** No unlearning method studied achieves
   forgetting by piggybacking the safety-refusal direction. Four converging
   negatives: geometry (no coherent per-question gate), safety-refusal encodes
   harmfulness and is silent on benign TOFU content, epistemic refusal can't be
   elicited (the model confabulates rather than abstaining), and the purest
   refusal direction aligns with unlearning-induced shifts no better than an
   arbitrary control direction.

2. **RMU's apparent activation-space "recovery" is a coherence artifact, not
   knowledge.** Difference-in-means ablation/translation restores fluent,
   on-topic generation but ~zero correct facts; the rising token-overlap
   "fact-hit" rate is schema-driven confabulation grazing a lenient metric — the
   same house-style fabrication the retain-oracle produces with no unlearning at
   all. The probe-normal direction is causally inert; only the difference-in-means
   direction is causal — separation is not mechanism.

3. **Latent vs destroyed (relearning speed).** Measuring how fast a few LoRA steps
   re-raise gold-answer probability, with a from-scratch control (invented authors)
   as the floor and the base model as the ceiling: RMU (representation-targeting)
   partially recovers, while AltPO (output-preference) does not recover under the
   same budget — output-preference unlearning is markedly more resistant to
   relearning. _(Knowledge-vs-coherence confirmation of the relearning signal via
   a generation-based factual eval is in progress; the log-prob result is
   validated by a flat-ceiling instrument check.)_

---

## Key notebooks

The investigation is organized as a sequence of notebooks. Four are central:

### `03_refusal_response_checks.ipynb` — extracting a refusal direction

Establishes how to elicit and extract a usable safety-refusal direction from the
model (harmful vs harmless prompts, dual-judge behavioral checks). Groundwork for
asking whether unlearning reuses this direction. Surfaces the key obstacle that
recurs throughout: behavioral elicitation is hard at 1B (the model confabulates
rather than abstaining), which blocks an _epistemic_ refusal direction the same way
it complicates the safety one.

### `08_ablation_translation.ipynb` — does ablation recover unlearned knowledge?

Tests whether removing/translating along candidate directions (probe-normal vs
difference-in-means) restores forgotten knowledge. Finding: the probe-normal
direction is causally inert; the difference-in-means direction _is_ causal and
drives RMU's gold-answer log-prob back toward oracle levels — but auditing the
generations shows the recovered text is fluent confabulation, not correct facts.
Demonstrates that token-overlap knowledge metrics are unreliable on TOFU.

### `11_refusal_alignment.ipynb` — do any methods piggyback safety refusal?

Directly tests refusal-piggybacking: builds the cleanest available refusal
direction and measures its alignment with each method's base->unlearned activation
shift, against a **refusal-free control direction** and a random floor. Decisive
result: the control direction aligns _at least as well_ as the refusal direction
for every method, so refusal is not special. Closes the project's original
question in the negative.

### `12_relearning.ipynb` — is forgotten knowledge latent or destroyed?

The project's only _write_ test (every other probe is a read test, which can't
distinguish "unreachable" from "gone"). Fine-tunes (LoRA) on forgotten facts and
measures the _rate_ gold-answer probability returns, against two controls:
from-scratch learning of **invented authors** (the destroyed-speed floor) and the
base model relearning forgotten facts (the ceiling). The ceiling doubles as a
validity gate — if fine-tuning degrades facts the model already knows, the
instrument is invalid. Result: RMU partially recovers; AltPO does not.

Supporting notebooks: `01_sanity_check` (pipeline), `02_magnitude_sweep`
(translation-magnitude sweep behind notebook 08's audit), `05_geometry_scan`,
`06_epistemic_refusal`.

---

## Methods under investigation

| Method                                       | Type                                | Relearning behavior (observed)                                |
| -------------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| RMU (Representation Misdirection Unlearning) | Representation targeting (layer 10) | Partially recovers under light fine-tuning (latent component) |
| AltPO                                        | Output preference                   | Does not recover under the same budget (more resistant)       |
| NPO / SimNPO / GradDiff                      | Output preference                   | Studied in refusal-alignment; not refusal-piggybacking        |

> The early prediction (RMU resists recovery, output-preference methods recover)
> was **inverted** by the results: RMU's representation-level edit is the _more_
> recoverable one, consistent with it being a localized activation mask over
> otherwise-intact weights.

---

## Model and data

- **Base (knows the facts):** `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- **Retain oracle (honest ignorance baseline):** `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`
- **Unlearned checkpoints:** `open-unlearning` on HuggingFace (forget10 split; RMU at layer 10, others at layer 14)
- **Dataset:** `locuslab/TOFU` — `forget10` (20 authors x 20 QA) and `retain90` (disjoint), GPT-4-generated fictitious-author biographies

---

## Setup

### Requirements

- Python 3.13, `uv` package manager
- ~55GB disk for checkpoints; 32GB RAM (Apple-Silicon MPS or CUDA)
- `peft` for the relearning (LoRA) experiments
- [Ollama](https://ollama.com) with `llama3.2` for behavioral / factual judging

### Install

```bash
uv sync
```

---

## Project structure

```
unlearning-probes/
├── notebooks/
│   ├── 01_sanity_check.ipynb
│   ├── 02_magnitude_sweep.ipynb
│   ├── 03_refusal_response_checks.ipynb   # extract refusal direction
│   ├── 05_geometry_scan.ipynb
│   ├── 06_epistemic_refusal.ipynb
│   ├── 08_ablation_translation.ipynb      # probe-normal vs diff-means recovery
│   ├── 11_refusal_alignment.ipynb         # refusal-piggyback test (vs control)
│   └── 12_relearning.ipynb                # latent-vs-destroyed (relearning rate)
├── src/
│   ├── config.py                          # checkpoints, per-method layers (single source of truth)
│   ├── model_loader.py                    # MPS-aware checkpoint loading
│   ├── intervention.py                    # diff-in-means, hooks, gold-answer scoring
│   ├── magnitude_sweep.py                 # translation-magnitude sweep + fact-hit
│   ├── refusal_alignment.py               # refusal direction + control + alignment
│   ├── invented_authors.py                # from-scratch control data (relearning)
│   ├── relearning.py                      # LoRA relearning loop + factual eval + judge
│   ├── audit_peak.py                      # generation audit at a sweep point
│   └── compare_relearning.py              # RMU-vs-AltPO comparison figure
├── results/relearning/                    # curves, logs (gitignored)
└── README.md
```

---

## Methodological notes (hard-won)

- **Always control for default structure.** Probe-normal separates classes but is
  causally inert (08); a naive Gaussian floor flagged every method until a
  refusal-free control was used (11); gold-answer log-prob conflates coherence with
  knowledge (02/08); the relearning ceiling-control catches an unstable instrument
  (12). The control is what tells you whether the measurement means anything.
- **Log-prob != knowledge.** Teacher-forced gold-answer probability rises with
  fluency/coherence, not only recall. Confirm knowledge claims with audited
  generations or a calibrated judge, never token-overlap alone.
- **One model resident at a time** on 32GB MPS; relearning saves per-checkpoint
  (crash-safe) with config-aware filenames (`{MODEL}.auth{N}.step{S}.lr{LR}...`)
  so distinct runs never overwrite each other.

---

## Status / next steps

- [x] Refusal-as-gate question — **closed** (not piggybacking; notebooks 03/05/06/11)
- [x] Ablation recovery — **coherence artifact**, audited on the full forget set (08/02)
- [x] Relearning log-prob result — RMU partial recovery vs AltPO resistance (12)
- [ ] Factual (knowledge-vs-coherence) confirmation of the relearning signal —
      generation + judge eval, instrument debugging in progress
- [ ] Geometry direction (post-talk): manifold curvature via Linear Field Probing
      on existing checkpoints; later, centroid-trajectory reconstruction across
      unlearning epochs (needs regenerated intermediate checkpoints)

---

## References

- Dower, S. — project problem statement (internal)
- Sarfati et al. (2026) — [The Shape of Beliefs](https://arxiv.org/abs/2602.02315)
- Fornasiere et al. (2026) — [Language Models Recognize Dropout and Gaussian Noise Applied to Their Activations](https://arxiv.org/abs/2604.17465)
- `open-unlearning` — TOFU unlearned checkpoints (HuggingFace)
- Maini et al. — TOFU benchmark
