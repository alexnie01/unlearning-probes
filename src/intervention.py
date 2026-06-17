"""
Activation intervention + target-answer log-prob scoring.

This module is the corrected replacement for the ad-hoc ablation/translation
code. It exists to answer one question precisely: when we remove or shift a
direction `d` from the residual stream at a given layer, does the model's
probability of the *correct* (forgotten) answer recover?

Design invariants (each fixes a specific bug in the old version):

  1. ALL-POSITION INTERVENTION. The hook edits the layer's output at every
     sequence position on the SAME forward pass that scores the answer.
     Causal attention means answer token i is computed by attending over the
     whole prefix (prompt + earlier answer tokens), so editing only the last
     prompt token leaks the un-edited signal into every prediction. We edit the
     full sequence-length dimension.

  2. TEACHER-FORCED ANSWER SCORING. We feed `prompt + answer` once and read the
     log-prob of answer token i from the logits at position i-1 (the standard
     off-by-one for next-token prediction). Prompt positions are masked out so
     only answer tokens contribute. We report the per-token MEAN to match the
     convention in baseline_logprobs.json.

  3. DTYPE/DEVICE DISCIPLINE. `d` arrives as float64 from sklearn/numpy; the
     model runs fp16 on MPS. The hook casts `d` to the activation's dtype and
     device, or the in-place edit silently no-ops.

  4. ABLATION vs TRANSLATION ARE DISTINCT.
       ablation:    h -> h - (h . d) d         (remove the component along d)
       translation: h -> h - c * d             (shift the mean along d by c)
     Each is run as its own scored pass against a shared baseline pass.

Typical use (from a notebook):

    from src.model_loader import load_model
    from src.intervention import (
        load_direction, score_dataset, run_recovery, recovery_table
    )

    model, tok, dev = load_model("open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    d = load_direction("../data/sweep_rmu/layer10/direction.npy")

    # c = forget_mean - retain_mean from the sweep metadata for this layer
    res = run_recovery(
        model, tok, dev,
        prompts=forget_questions, answers=forget_answers,
        layer_name="model.layers.10",
        direction=d, translate_c=gap,
    )
    recovery_table(res)            # mean baseline / ablated / translated
    res["per_question"]            # inspect the deltas question-by-question
"""

import json
from contextlib import contextmanager

import numpy as np
import torch

from src.hooks import extract_activations


# ---------------------------------------------------------------------------
# Direction loading
# ---------------------------------------------------------------------------

def load_direction(path: str) -> np.ndarray:
    """
    Load a unit-norm direction saved by layer_sweep.py (direction.npy).

    Re-normalizes defensively so the ablation projection h-(h.d)d is exact even
    if the saved vector drifted from unit norm.
    """
    d = np.load(path)
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError(f"Direction at {path} has zero norm.")
    return d / norm


def diff_in_means_direction(
    model, tokenizer, device,
    forget_prompts, retain_prompts,
    layer_name: str,
    return_raw_gap: bool = False,
):
    """
    Compute the difference-in-means direction at a layer: the unit vector from
    the retain-set activation mean to the forget-set activation mean.

        d = (mean(forget_acts) - mean(retain_acts)) / ||.||

    This is an ALTERNATIVE to the probe-normal direction (load_direction). The
    probe normal is a *classifier* boundary — it can exploit any feature that
    discriminates forget from retain, including features that merely correlate
    with membership rather than causing suppression. Difference-in-means is just
    where the two activation clouds' centers sit. Ablating BOTH and finding both
    inert shows the result isn't an artifact of which separating direction you
    chose — it makes "separation != causal mechanism" robust rather than
    probe-specific.

    Args:
        forget_prompts, retain_prompts: question lists (retain is truncated to
            len(forget) for a balanced mean).
        layer_name: e.g. "model.layers.14".
        return_raw_gap: if True, also return the projection gap in RAW space
            (forget_mean_proj - retain_mean_proj along the unit direction),
            usable as translate_c for a translation experiment.

    Returns:
        d (unit-norm np.ndarray), or (d, raw_gap) if return_raw_gap.
    """
    n = len(forget_prompts)
    f_acts = extract_activations(model, tokenizer, forget_prompts, layer_name, device)
    r_acts = extract_activations(model, tokenizer, retain_prompts[:n], layer_name, device)

    raw_dir = f_acts.mean(axis=0) - r_acts.mean(axis=0)
    norm = np.linalg.norm(raw_dir)
    if norm == 0:
        raise ValueError("Difference-in-means direction has zero norm "
                         "(forget and retain means coincide at this layer).")
    d = raw_dir / norm

    if return_raw_gap:
        raw_gap = float(f_acts.mean(axis=0) @ d - r_acts.mean(axis=0) @ d)
        return d, raw_gap
    return d


def gap_from_metadata(metadata_path: str, layer: int) -> float:
    """
    Read the data-derived translation distance c = forget_mean - retain_mean
    for a given layer from a sweep's sweep_metadata.json.

    This is the principled translation magnitude: it moves forget-set
    activations along d to where the retain-set activations sit on that axis.
    """
    with open(metadata_path) as f:
        meta = json.load(f)
    m = meta["metrics"][str(layer)]
    return float(m["forget_mean"] - m["retain_mean"])


# ---------------------------------------------------------------------------
# Intervention hook
# ---------------------------------------------------------------------------

@contextmanager
def _intervention_hook(model, layer_name: str, fn):
    """
    Register a forward hook on `layer_name` that replaces the module's output
    with fn(output) for the duration of the context, then removes it.

    `fn` receives and returns the hidden-state tensor (batch, seq, hidden).
    Handles modules that return a tuple (hidden, *rest) by editing element 0.
    """
    module = dict(model.named_modules())[layer_name]

    def _hook(_mod, _inp, output):
        # IMPORTANT: modify IN-PLACE. transformers >=5.x decoder layers return a
        # bare tensor and the modeling code captures it by reference before a
        # hook's *returned* replacement would propagate — so a return-based edit
        # is silently discarded. Mutating the tensor object itself always
        # propagates because downstream code holds the same reference.
        hidden = output[0] if isinstance(output, tuple) else output
        fn(hidden)               # fn mutates `hidden` in place; no return used
        return output

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _make_ablation_fn(direction: np.ndarray):
    """In-place fn: h <- h - (h . d) d, applied at ALL positions."""
    def fn(hidden: torch.Tensor) -> None:
        d = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
        # hidden: (batch, seq, hidden_size); d: (hidden_size,)
        proj = torch.matmul(hidden, d)              # (batch, seq)
        hidden.sub_(proj.unsqueeze(-1) * d)          # in-place subtract
    return fn


def _make_translation_fn(direction: np.ndarray, c: float):
    """In-place fn: h <- h - c * d, applied at ALL positions."""
    def fn(hidden: torch.Tensor) -> None:
        d = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden.sub_(c * d)                           # in-place subtract
    return fn


# ---------------------------------------------------------------------------
# Target-answer log-prob scoring
# ---------------------------------------------------------------------------

def _score_one(model, tokenizer, device, prompt: str, answer: str) -> float:
    """
    Teacher-forced mean log-prob of `answer` tokens given `prompt`.

    Tokenizes prompt and answer separately to find the boundary, concatenates,
    runs ONE forward pass, and averages the log-probs of the answer tokens
    using the off-by-one (logits at i-1 predict token i).

    Assumes whatever intervention hook is active is already registered by the
    caller, so this single pass reflects the intervened residual stream.
    """
    # Tokenize separately to locate the answer span. add_special_tokens only on
    # the prompt so we don't inject a BOS in the middle.
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    answer_ids = tokenizer(
        answer, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits            # (1, seq, vocab)

    log_probs = torch.log_softmax(logits.float(), dim=-1)

    n_prompt = prompt_ids.shape[1]
    n_answer = answer_ids.shape[1]
    total = 0.0
    for i in range(n_answer):
        # answer token sits at absolute position (n_prompt + i);
        # it is predicted by the logits at position (n_prompt + i - 1)
        tok = input_ids[0, n_prompt + i]
        lp = log_probs[0, n_prompt + i - 1, tok]
        total += lp.item()
    return total / n_answer                          # per-token MEAN


def score_dataset(
    model, tokenizer, device,
    prompts, answers,
    layer_name=None, intervention_fn=None,
) -> np.ndarray:
    """
    Score every (prompt, answer) pair, optionally under an intervention.

    If layer_name and intervention_fn are both given, the hook is active for
    the whole scoring loop. If either is None, this scores the clean baseline.

    Returns: np.ndarray of per-question mean log-probs, shape (n,).
    """
    assert len(prompts) == len(answers), "prompts/answers length mismatch"

    def _loop():
        return np.array([
            _score_one(model, tokenizer, device, p, a)
            for p, a in zip(prompts, answers)
        ])

    if layer_name is not None and intervention_fn is not None:
        with _intervention_hook(model, layer_name, intervention_fn):
            return _loop()
    return _loop()


# ---------------------------------------------------------------------------
# Full recovery experiment
# ---------------------------------------------------------------------------

def run_recovery(
    model, tokenizer, device,
    prompts, answers,
    layer_name: str,
    direction: np.ndarray,
    translate_c: float,
) -> dict:
    """
    Run baseline, ablation, and translation scoring on the same data and layer.

    Three passes:
      baseline    — no hook
      ablated     — h - (h.d) d   at all positions of `layer_name`
      translated  — h - c * d     at all positions of `layer_name`

    Returns a dict with per-question arrays, means, and signed deltas. The
    deltas are what actually matter: a real linear gate shows LARGE, UNEVEN
    per-question recovery. A uniform shift across all questions (per-question
    delta roughly constant) is an offset removal, not knowledge recovery.
    """
    baseline = score_dataset(model, tokenizer, device, prompts, answers)

    abl_fn = _make_ablation_fn(direction)
    ablated = score_dataset(
        model, tokenizer, device, prompts, answers,
        layer_name=layer_name, intervention_fn=abl_fn,
    )

    tr_fn = _make_translation_fn(direction, translate_c)
    translated = score_dataset(
        model, tokenizer, device, prompts, answers,
        layer_name=layer_name, intervention_fn=tr_fn,
    )

    d_abl = ablated - baseline
    d_tr = translated - baseline

    return {
        "layer_name": layer_name,
        "translate_c": translate_c,
        "baseline_lps": baseline,
        "ablated_lps": ablated,
        "translated_lps": translated,
        "mean_baseline": float(baseline.mean()),
        "mean_ablated": float(ablated.mean()),
        "mean_translated": float(translated.mean()),
        "per_question": {
            "delta_ablation": d_abl,
            "delta_translation": d_tr,
        },
        # Diagnostics that separate "offset removal" from "real recovery":
        # if mean|delta| approx |mean delta|, the effect is a uniform shift.
        "ablation_mean_signed_delta": float(d_abl.mean()),
        "ablation_mean_abs_delta": float(np.abs(d_abl).mean()),
        "translation_mean_signed_delta": float(d_tr.mean()),
        "translation_mean_abs_delta": float(np.abs(d_tr).mean()),
    }


def recovery_table(res: dict) -> None:
    """Pretty-print the means and the uniform-shift diagnostic for one run."""
    print(f"layer: {res['layer_name']}   c = {res['translate_c']:+.4f}")
    print(f"  mean baseline    : {res['mean_baseline']:+.4f}")
    print(f"  mean ablated     : {res['mean_ablated']:+.4f}  "
          f"(delta {res['mean_ablated'] - res['mean_baseline']:+.4f})")
    print(f"  mean translated  : {res['mean_translated']:+.4f}  "
          f"(delta {res['mean_translated'] - res['mean_baseline']:+.4f})")
    print("  uniform-shift check (ablation):")
    print(f"    signed mean delta = {res['ablation_mean_signed_delta']:+.4f}")
    print(f"    abs    mean delta = {res['ablation_mean_abs_delta']:.4f}")
    print("    -> if these two are nearly equal, the ablation is a uniform")
    print("       offset removal, NOT selective knowledge recovery.")