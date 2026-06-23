"""
Cosine alignment between the v3 refusal direction and each unlearning method's
mean activation-shift, benchmarked against the random-vector floor.

WHY THIS DESIGN (and not a bare cosine):
  A bare cosine ~ 0 is over-determined — in 2048 dims any two mechanistically
  unrelated directions are near-orthogonal by default, AND v3 (a within-harmful
  refusal direction) has little signal on benign TOFU. So "~0" could mean "no
  piggybacking", "v3 silent on benign data", or "random-floor orthogonality" —
  three confounded readings. The informative question is therefore NOT "is it
  zero" but "is it ABOVE THE RANDOM FLOOR". Comparing to the floor controls for
  the dimensionality explanation and converts an ambiguous zero into a
  defensible "no more than chance" (same move that made centered-coherence ~0
  meaningful via its random baseline).

WHAT WE TAKE THE COSINE *WITH*:
  The geometry result says the base->unlearned shift has NO coherent per-question
  direction (centered coherence ~0): it is dominated by a constant MEAN OFFSET.
  So the only well-defined target is that mean-offset vector per method:
      offset_m = mean_q( act_unlearned_m[q] - act_base[q] )   at a fixed layer.
  Cosine against per-question residuals would be meaningless (they are ~random).

LAYER CHOICE:
  v3 lives wherever the refusal decision is most legible — NOT necessarily RMU's
  layer 10. Pass the layer you extracted v3 at; the offsets are computed at the
  SAME layer so the cosine is between two vectors in one shared space.

Typical use (build v3 from the labeled CSV, in one base-model residency):
    from src.config import BASE_MODEL, CHECKPOINTS, checkpoint
    from src.refusal_alignment import run_alignment, alignment_table, plot_alignment
    res = run_alignment(
        layer=14,
        base_model_id=BASE_MODEL,
        method_ids={m: checkpoint(m) for m in CHECKPOINTS},
        forget_prompts=forget_prompts,
        labeled_csv="../results/cleaned_harmful_harmless_responses_labeled.csv",
    )
    alignment_table(res)         # text table vs the chance floor
    plot_alignment(res)          # figure -> results/alignment_refusal/

(Or pass a precomputed v3_direction=<unit vector at `layer`> instead of
labeled_csv if you already have the direction.)
"""

import gc
import numpy as np
import pandas as pd
import torch

from src.model_loader import load_model
from src.hooks import extract_activations
from src.geometry import random_coherence_floor   # reused random baseline


# ---------------------------------------------------------------------------
# Build v3 (within-harmful refusal direction) from the labeled responses CSV
# ---------------------------------------------------------------------------

def build_v3(
    model, tokenizer, device,
    labeled_csv: str,
    layer: int,
    refused_col: str = "refused",
    type_col: str = "type",
    prompt_col: str = "prompt",
) -> tuple[np.ndarray, dict]:
    """
    Reconstruct the v3 refusal direction = mean(harmful-refused activations)
    - mean(harmful-complied activations), unit-normalized, at `layer`.

    v3 is the PUREST refusal contrast: both centroids are harmful prompts, so
    the only thing varying between them is the refusal DECISION, not harmfulness.
    It is an honest difference-in-means (not a probe normal), so it is the
    causal-grade construction — do NOT substitute a saved probe direction.npy,
    which is a classifier boundary (Phase-1 Finding #1: separates but not causal).

    The model passed in MUST be the model whose refusal direction you want —
    normally the BASE full model (the refusal mechanism you're testing unlearning
    against), at the SAME layer you will run the alignment at.

    Returns:
        v3 (unit np.ndarray), info dict with n_refused / n_complied and a
        stability flag. Warns if either class is small (a 15-example centroid in
        2048-d is noisy — your past finding).
    """
    layer_name = f"model.layers.{layer}"
    df = pd.read_csv(labeled_csv)

    def _is_true(x):
        return str(x).strip().lower() in ("true", "1", "1.0")

    harmful = df[df[type_col] == "harmful"]
    refused  = harmful[harmful[refused_col].map(_is_true)]
    complied = harmful[~harmful[refused_col].map(_is_true)]

    n_ref, n_comp = len(refused), len(complied)
    if n_ref < 2 or n_comp < 2:
        raise ValueError(
            f"Need >=2 examples per class for a mean; got refused={n_ref}, "
            f"complied={n_comp}."
        )

    ref_acts  = extract_activations(model, tokenizer, refused[prompt_col].tolist(),
                                    layer_name, device)
    comp_acts = extract_activations(model, tokenizer, complied[prompt_col].tolist(),
                                    layer_name, device)

    raw = ref_acts.mean(axis=0) - comp_acts.mean(axis=0)
    v3 = _unit(raw.astype(np.float64))

    small = min(n_ref, n_comp) < 20
    info = {
        "layer": layer,
        "n_refused": n_ref,
        "n_complied": n_comp,
        "raw_norm": float(np.linalg.norm(raw)),
        "small_class_warning": small,
    }
    print(f"v3 built at layer {layer}: refused={n_ref}, complied={n_comp}, "
          f"raw_norm={info['raw_norm']:.3f}")
    if small:
        print(f"  WARNING: smallest class = {min(n_ref, n_comp)} examples. A "
              f"centroid from <20 points in 2048-d is noisy; treat any single "
              f"above-floor result with suspicion and re-check across seeds.")
    return v3, info


def build_control_direction(
    model, tokenizer, device,
    labeled_csv: str,
    layer: int,
    type_col: str = "type",
    prompt_col: str = "prompt",
    seed: int = 0,
) -> np.ndarray:
    """
    A "nothing" control direction: difference-in-means between two RANDOM halves
    of the HARMLESS prompts, at `layer`, unit-normalized.

    Purpose — isolate the offset-structure confound. v3 and this control share
    everything incidental (dimensionality, living in the activation manifold,
    any component the unlearning offsets share by default) but the control has
    NO refusal signal (it's an arbitrary split of benign prompts). So if the
    control scores cosines against the unlearning offsets as high as v3 does,
    v3's "above floor" verdicts are NOT refusal — they're the shared structure
    any activation-space direction picks up. The control is the honest chance
    level for THIS comparison; the random-Gaussian floor is too permissive
    because real offsets are not isotropic.
    """
    layer_name = f"model.layers.{layer}"
    df = pd.read_csv(labeled_csv)
    harmless = df[df[type_col] == "harmless"][prompt_col].tolist()
    if len(harmless) < 4:
        raise ValueError(f"need >=4 harmless prompts to split; got {len(harmless)}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(harmless))
    half = len(idx) // 2
    a = [harmless[i] for i in idx[:half]]
    b = [harmless[i] for i in idx[half:2 * half]]   # balanced halves

    a_acts = extract_activations(model, tokenizer, a, layer_name, device)
    b_acts = extract_activations(model, tokenizer, b, layer_name, device)
    raw = a_acts.mean(axis=0) - b_acts.mean(axis=0)
    ctrl = _unit(raw.astype(np.float64))
    print(f"control direction built at layer {layer}: "
          f"{len(a)} vs {len(b)} harmless prompts (no refusal signal by design)")
    return ctrl


def _free():
    """Reclaim MPS memory. Call AFTER `del`-ing the caller's own model/tok
    locals — see run_alignment. The actual freeing comes from the caller's `del`
    (dropping the binding that holds the model); this just forces collection and
    clears the Metal cache so the memory is returned promptly rather than lazily.

    Why not `_free(model)`? Passing the object and `del`-ing the parameter does
    NOT free it — the caller's local still references it (same trap as the
    notebook free_model bug). The binding that must be dropped lives in the
    caller, so the caller deletes it; this helper only does the gc/cache step."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-norm vector")
    return v / n


def mean_offset(base_acts: np.ndarray, tgt_acts: np.ndarray) -> np.ndarray:
    """Constant component of the base->target shift: mean over questions of the
    per-question difference. Rows of both arrays are aligned (same prompts,
    same order). Returns the RAW (un-normalized) offset vector."""
    assert base_acts.shape == tgt_acts.shape, "activation arrays misaligned"
    return (tgt_acts - base_acts).mean(axis=0)


def random_floor_cosine(
    direction: np.ndarray, n_samples: int = 2000, seed: int = 42
) -> dict:
    """Distribution of |cosine| between `direction` and random unit vectors of
    the same dimension. This is the chance baseline a real alignment must beat.
    Returns mean and a high percentile (the 'floor' a signal must exceed)."""
    rng = np.random.default_rng(seed)
    d = _unit(direction)
    rand = rng.standard_normal((n_samples, d.shape[0]))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    cos = np.abs(rand @ d)
    return {
        "mean_abs_cos": float(cos.mean()),
        "p95_abs_cos":  float(np.percentile(cos, 95)),
        "p99_abs_cos":  float(np.percentile(cos, 99)),
    }


def run_alignment(
    layer: int,
    base_model_id: str,
    method_ids: dict[str, str],
    forget_prompts: list[str],
    v3_direction: np.ndarray | None = None,
    labeled_csv: str | None = None,
) -> dict:
    """
    Cosine(v3, mean-offset_method) for each method, at `layer`, vs random floor.

    v3 source — provide EXACTLY ONE of:
      v3_direction : a precomputed unit vector at `layer` (if you already have it)
      labeled_csv  : path to the cleaned refusal CSV; v3 is built on the BASE
                     model during the same residency used for base activations
                     (one base load does both jobs — efficient and ensures v3 and
                     the base reference come from the same model, as they must).

    Loads the base model once (build v3 if needed + extract forget activations),
    frees it, then loads each method once (extract the same prompts' activations,
    compute the offset), frees it. One model resident at a time -> M1-safe.

    Returns dict with the random floor, the v3 build info (if built here), and
    per method the raw cosine and 'above_floor' (|cos| > p99 of the floor).
    """
    if (v3_direction is None) == (labeled_csv is None):
        raise ValueError("Provide exactly one of v3_direction or labeled_csv.")

    layer_name = f"model.layers.{layer}"

    # Single base residency: build v3 (if from CSV), the control direction, AND
    # extract base activations — all while the base model is resident once.
    base_model, tok, dev = load_model(base_model_id)
    v3_info = None
    ctrl = None
    if labeled_csv is not None:
        v3, v3_info = build_v3(base_model, tok, dev, labeled_csv, layer)
        ctrl = build_control_direction(base_model, tok, dev, labeled_csv, layer)
    else:
        v3 = _unit(np.asarray(v3_direction, dtype=np.float64))
    base_acts = extract_activations(base_model, tok, forget_prompts, layer_name, dev)
    del base_model, tok          # drop the caller's references BEFORE gc
    _free()

    floor = random_floor_cosine(v3)

    results = {}
    for label, path in method_ids.items():
        model, tok, dev = load_model(path)
        tgt_acts = extract_activations(model, tok, forget_prompts, layer_name, dev)
        del model, tok           # drop refs now, so the NEXT load_model peaks at
        _free()                  # one model, not two (old + new) simultaneously

        offset = mean_offset(base_acts, tgt_acts)
        offset_u = _unit(offset)
        cos = float(np.dot(v3, offset_u))          # signed cosine vs v3
        cos_ctrl = float(np.dot(ctrl, offset_u)) if ctrl is not None else None
        results[label] = {
            "cosine": cos,
            "abs_cosine": abs(cos),
            "cosine_control": cos_ctrl,
            "abs_cosine_control": abs(cos_ctrl) if cos_ctrl is not None else None,
            "offset_norm": float(np.linalg.norm(offset)),
            "above_floor": abs(cos) > floor["p99_abs_cos"],
            # The decisive test: is v3 meaningfully ABOVE the control? If not,
            # the "above floor" verdict is the shared-structure confound.
            "above_control": (abs(cos) > abs(cos_ctrl)) if cos_ctrl is not None else None,
        }
        ctrl_str = f" ctrl={cos_ctrl:+.3f}" if cos_ctrl is not None else ""
        print(f"{label:9} cos={cos:+.3f} |cos|={abs(cos):.3f}{ctrl_str} "
              f"{'ABOVE floor' if results[label]['above_floor'] else 'at floor'}")

    return {
        "layer": layer,
        "random_floor": floor,
        "v3_info": v3_info,
        "has_control": ctrl is not None,
        "methods": results,
    }


def alignment_table(res: dict) -> None:
    """Print the result with the CONTROL comparison as the real verdict.

    'above floor' (vs random-Gaussian) is now known to be too permissive —
    real offsets aren't isotropic, so any activation-space direction can clear
    it. The honest question is whether v3 beats a refusal-free CONTROL direction
    against the same offsets. If v3 ~ control, the alignment is the shared-
    structure confound, not refusal."""
    f = res["random_floor"]
    has_ctrl = res.get("has_control", False)
    print(f"\nLayer {res['layer']}  —  v3 refusal direction vs unlearning mean-offset")
    print(f"random-Gaussian floor (too permissive): mean|cos|={f['mean_abs_cos']:.3f}  "
          f"p99={f['p99_abs_cos']:.3f}")
    print("-" * 72)
    if has_ctrl:
        print(f"{'method':10}{'v3 cos':>9}{'|v3|':>8}{'ctrl cos':>10}{'|ctrl|':>8}"
              f"{'verdict':>16}")
    else:
        print(f"{'method':10}{'cosine':>10}{'|cosine|':>10}{'verdict':>16}")

    signs = []
    for label, m in res["methods"].items():
        signs.append(m["cosine"] > 0)
        if has_ctrl:
            # verdict: does v3 meaningfully beat the control? require a margin,
            # not a hair, so noise in either direction doesn't flip it.
            beats = m["abs_cosine"] > m["abs_cosine_control"] * 1.5
            verdict = "v3 > control" if beats else "= control"
            print(f"{label:10}{m['cosine']:>+9.3f}{m['abs_cosine']:>8.3f}"
                  f"{m['cosine_control']:>+10.3f}{m['abs_cosine_control']:>8.3f}"
                  f"{verdict:>16}")
        else:
            verdict = "ABOVE FLOOR" if m["above_floor"] else "at chance"
            print(f"{label:10}{m['cosine']:>+10.3f}{m['abs_cosine']:>10.3f}{verdict:>16}")

    print("-" * 72)

    # Sign diagnostic: random overlaps are sign-symmetric (~half negative).
    # All-positive across methods => a shared component pulls every offset the
    # same way relative to v3 => evidence of the structural confound.
    n_pos = sum(signs)
    n = len(signs)
    print(f"sign check: {n_pos}/{n} cosines positive.", end=" ")
    if n_pos == n or n_pos == 0:
        print("ALL same sign -> shared offset component (confound signal).")
    else:
        print("mixed signs -> no obvious shared-component artifact.")

    if has_ctrl:
        any_beats = any(
            m["abs_cosine"] > m["abs_cosine_control"] * 1.5
            for m in res["methods"].values()
        )
        if any_beats:
            print("=> Some method's v3 alignment EXCEEDS the refusal-free control"
                  " by a margin -> worth investigating that method specifically.")
        else:
            print("=> No method's v3 alignment beats the refusal-free control.")
            print("   The 'above floor' verdicts were the shared-structure confound,"
                  " not refusal.")
            print("   Clean negative: v3 is no more aligned with unlearning shifts"
                  " than an arbitrary direction.")
    else:
        any_above = any(m["above_floor"] for m in res["methods"].values())
        print("=> (no control direction run; rerun with labeled_csv for the"
              " decisive test)")


def plot_alignment(res: dict, save_dir: str = "../results/alignment_refusal",
                   filename: str = "v3_vs_unlearning_alignment.png", ax=None):
    """
    The result figure: each method's |cosine(v3, mean-offset)| against the
    RANDOM-FLOOR band. The band is the load-bearing element — without it, small
    cosines read as "weak effect" when they are actually chance. Methods inside
    the band = no above-random alignment (the expected clean negative); a method
    above the band stands out and demands a second look.

    Saves a talk-ready PNG to save_dir (a RESULT, not intermediate data). Pass
    save_dir=None to skip saving.
    """
    import os
    import matplotlib.pyplot as plt

    f = res["random_floor"]
    has_ctrl = res.get("has_control", False)
    labels = list(res["methods"].keys())
    abs_cos = [res["methods"][m]["abs_cosine"] for m in labels]
    abs_ctrl = [res["methods"][m]["abs_cosine_control"] for m in labels] if has_ctrl else None
    x = np.arange(len(labels))

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    # Floor band: mean to p99 shaded. The permissive null, shown for reference.
    ax.axhspan(f["mean_abs_cos"], f["p99_abs_cos"], color="grey", alpha=0.18,
               label="random-Gaussian floor (too permissive)")
    ax.axhline(f["p99_abs_cos"], ls="--", color="grey", lw=1)

    if has_ctrl:
        w = 0.38
        ax.bar(x - w/2, abs_cos, width=w, color="tab:blue", edgecolor="black",
               lw=0.6, label="v3 (refusal)")
        ax.bar(x + w/2, abs_ctrl, width=w, color="tab:orange", edgecolor="black",
               lw=0.6, label="control (refusal-free)")
        title2 = "v3 vs a refusal-FREE control against the same offsets:\n" \
                 "if the bars match, the alignment is shared structure, not refusal"
    else:
        above = [res["methods"][m]["above_floor"] for m in labels]
        colors = ["tab:red" if a else "tab:blue" for a in above]
        ax.bar(x, abs_cos, color=colors, edgecolor="black", lw=0.6, width=0.6)
        title2 = "bars inside the grey band are indistinguishable from chance"

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|cosine with unlearning mean-offset|")
    ax.set_title(f"Refusal-direction alignment with unlearning shift  "
                 f"(layer {res['layer']})\n{title2}")
    ymax = max(abs_cos + (abs_ctrl if abs_ctrl else []) + [f["p99_abs_cos"]])
    ax.set_ylim(0, ymax * 1.3 + 1e-6)
    ax.legend(fontsize=8, loc="upper left")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved figure to {path}")
    return ax


if __name__ == "__main__":
    # Logic smoke test (no models): an offset deliberately built to align with
    # v3 must read ABOVE floor; an orthogonal one must read at chance.
    rng = np.random.default_rng(0)
    dim = 2048
    v3 = _unit(rng.standard_normal(dim))
    floor = random_floor_cosine(v3)

    aligned = _unit(v3 + 0.05 * rng.standard_normal(dim))   # nearly parallel
    orth = rng.standard_normal(dim)
    orth = _unit(orth - (orth @ v3) * v3)                   # exactly orthogonal

    a = abs(float(v3 @ aligned))
    o = abs(float(v3 @ orth))
    assert a > floor["p99_abs_cos"], "aligned vector should beat the floor"
    assert o <= floor["p99_abs_cos"], "orthogonal vector should sit at floor"
    print(f"smoke OK: aligned|cos|={a:.3f} > p99={floor['p99_abs_cos']:.3f} > "
          f"orth|cos|={o:.3f}")