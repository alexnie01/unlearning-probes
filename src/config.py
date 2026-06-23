"""
Single source of truth for model identities and intervention sites.

Import this everywhere instead of pasting the dicts into each notebook, so all
notebooks/scripts agree on exactly which checkpoint "RMU" is and which layer to
intervene at. This module is INERT on import: no path resolution, no directory
creation, no I/O. Output paths and makedirs belong to the driver (notebook or
script) that knows where it is being run from.

Why layers are stored as module-name strings (not ints): the harness
(intervention.py, geometry.py) consumes layer names like "model.layers.10".
Storing the string here means callers never re-derive it (and never disagree on
the f-string). Use layer_name(label) to fetch it.
"""

# --- Model identities (invariant: same everywhere, regardless of caller) ------

BASE_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
RETAIN     = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"

CHECKPOINTS = {
    "GradDiff": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5",
    "NPO":      "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10",
    "AltPO":    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10",
    "SimNPO":   "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.125_ep10",
    "RMU":      "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer10_scoeff10_epoch10",
}

# --- Intervention site per model ---------------------------------------------
# RMU was TRAINED at layer 10 (its checkpoint name encodes layer10), so that is
# the principled site to intervene at for RMU. The output-preference methods
# (GradDiff/NPO/AltPO/SimNPO) have no designated layer; 14 is the project
# default from the Phase-1 sweep. Stored as ints here; use layer_name() for the
# module-name string the hooks expect.

_DEFAULT_LAYER = 14
MODEL_LAYER_INT = {label: _DEFAULT_LAYER for label in CHECKPOINTS}
MODEL_LAYER_INT["RMU"] = 10


def layer_name(label: str) -> str:
    """Module-name string for `label`'s intervention site, e.g. 'model.layers.10'.

    Raises KeyError with the known labels if `label` is unrecognised, so a typo
    fails loudly instead of silently defaulting.
    """
    if label not in MODEL_LAYER_INT:
        raise KeyError(
            f"Unknown model label {label!r}. Known: {sorted(MODEL_LAYER_INT)}"
        )
    return f"model.layers.{MODEL_LAYER_INT[label]}"


def checkpoint(label: str) -> str:
    """HF path for `label`, with a loud error on typos."""
    if label not in CHECKPOINTS:
        raise KeyError(
            f"Unknown model label {label!r}. Known: {sorted(CHECKPOINTS)}"
        )
    return CHECKPOINTS[label]


# Convenience: live cases vs controls, per Phase-1 findings. AltPO/RMU were the
# genuinely-suppressed ones; the rest barely moved and act as controls.
LIVE_CASES = ["AltPO", "RMU"]
CONTROLS   = ["GradDiff", "NPO", "SimNPO"]


if __name__ == "__main__":
    # Inertness + accessor smoke test (no model, no I/O).
    assert layer_name("RMU") == "model.layers.10"
    assert layer_name("NPO") == "model.layers.14"
    assert checkpoint("RMU").endswith("RMU_lr5e-05_layer10_scoeff10_epoch10")
    assert set(LIVE_CASES) | set(CONTROLS) == set(CHECKPOINTS)
    try:
        layer_name("rmu")  # wrong case -> should raise
        raise AssertionError("expected KeyError on bad label")
    except KeyError:
        pass
    print("config smoke test OK:",
          {k: layer_name(k) for k in CHECKPOINTS})