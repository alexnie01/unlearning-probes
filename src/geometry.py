"""
Geometry analysis for unlearning transformations.

Measures how unlearning moves activations relative to a base model, across
all layers, for a set of checkpoints. For each (checkpoint, question_set, layer)
it reports:

    coherence    — mean pairwise cosine similarity of per-question shift vectors
                   (dominated by the shared mean offset)
    centered     — same, after removing the mean shift; ~0 means no shared
                   per-question direction beyond the offset
    shift_norm   — magnitude of the mean offset (the constant component)
    resid_norm   — typical magnitude of per-question residual movement

Each model is loaded exactly once and all requested question sets are extracted
while it is in memory, avoiding redundant loads. This matters most at larger
model sizes where loading dominates runtime.
"""

import gc
import numpy as np
import torch

from src.model_loader import load_model
from src.hooks import extract_activations_multi


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def coherence(diffs: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity among row vectors of `diffs`.

    Rows with near-zero norm are dropped (e.g. RMU below its intervention
    layer, where activations are unchanged). Returns NaN if fewer than two
    valid rows remain.
    """
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    valid = norms.squeeze() > 1e-8
    if valid.sum() < 2:
        return float("nan")
    units = diffs[valid] / norms[valid]
    cos = units @ units.T
    off = cos[~np.eye(int(valid.sum()), dtype=bool)]
    return float(off.mean())


def random_coherence_floor(n: int, dim: int, seed: int = 42) -> float:
    """Expected coherence for n random unit vectors in `dim` dimensions."""
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((n, dim))
    units = rand / np.linalg.norm(rand, axis=1, keepdims=True)
    off = (units @ units.T)[~np.eye(n, dtype=bool)]
    return float(off.mean())


def shift_geometry(base_acts: dict, tgt_acts: dict, layers: list[str]) -> dict:
    """
    Per-layer shift geometry between a base and target activation set.
    Both dicts map layer_name -> array of shape (n_questions, hidden_size),
    aligned by row (same questions in the same order).
    """
    out = {}
    for layer in layers:
        diffs = tgt_acts[layer] - base_acts[layer]
        mean_shift = diffs.mean(axis=0)
        residuals = diffs - mean_shift
        out[layer] = {
            "coherence":  coherence(diffs),
            "centered":   coherence(residuals),
            "shift_norm": float(np.linalg.norm(mean_shift)),
            "resid_norm": float(np.linalg.norm(residuals, axis=1).mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _free(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _extract_all_sets(model, tokenizer, device, question_sets, layers):
    """Extract every question set during a single model residency in memory."""
    return {
        set_name: extract_activations_multi(model, tokenizer, questions, layers, device)
        for set_name, questions in question_sets.items()
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_geometry_scan(
    base_model_id: str,
    targets: dict[str, str],
    question_sets: dict[str, list[str]],
    layers: list[str] | None = None,
):
    """
    Scan shift geometry for each target checkpoint against a base model,
    across one or more named question sets.

    Each model (base + every target) is loaded exactly once; all question
    sets are extracted while that model is resident, then it is freed.

    Args:
        base_model_id: HF path of the base (pre-unlearning) model
        targets:       {label: hf_path} for each checkpoint to compare to base
        question_sets: {set_name: [questions]} — e.g.
                       {"forget": forget_qs, "retain": retain_qs}
                       Accepts any number of named sets (1, 2, 3+).
        layers:        layer module names; defaults to all 16

    Returns:
        scan: nested dict
              scan[set_name][label][layer] -> {coherence, centered,
                                               shift_norm, resid_norm}
    """
    if layers is None:
        layers = [f"model.layers.{i}" for i in range(16)]

    # Base model: one load, extract every question set
    base_model, tokenizer, device = load_model(base_model_id)
    base_acts = _extract_all_sets(base_model, tokenizer, device, question_sets, layers)
    _free(base_model)

    # Initialise nested result structure: scan[set_name][label]
    scan = {set_name: {} for set_name in question_sets}

    for label, path in targets.items():
        model, tok, dev = load_model(path)
        tgt_acts = _extract_all_sets(model, tok, dev, question_sets, layers)
        _free(model)

        for set_name in question_sets:
            scan[set_name][label] = shift_geometry(
                base_acts[set_name], tgt_acts[set_name], layers
            )
        print(f"done: {label}")

    return scan