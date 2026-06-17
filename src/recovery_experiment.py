"""
Recovery-experiment runner across unlearning methods.

Separates three concerns:
  - run_one_model:   load a checkpoint, run the recovery experiment, return a
                     plain dict of results + diagnostics (NO printing/plotting).
  - run_all_models:  loop run_one_model over a set of checkpoints, collect
                     everything into one structure, optionally save to JSON.
  - summarize / plot helpers: consume the collected structure for display.

The split means the expensive model passes happen once and produce data you
can re-summarize, re-plot, and archive without rerunning anything.

Reference points for interpretation (per the project):
  base model ~ knows everything; retain oracle ~ honest ignorance.
  "Recovery" means a suppressed model's forget log-prob climbing back TOWARD
  the oracle (or past it), not toward 0. A uniform per-question shift
  (signed delta ~ abs delta) is offset removal, not selective recovery.
"""

import json
import os

import numpy as np

from src.model_loader import load_model
from src.intervention import (
    load_direction, gap_from_metadata,
    score_dataset, run_recovery, _make_ablation_fn,
)


def best_layer_from_metadata(meta_path: str) -> int:
    """Return the layer index with highest probe test accuracy from a sweep."""
    with open(meta_path) as f:
        meta = json.load(f)
    metrics = meta["metrics"]
    return int(max(metrics, key=lambda k: metrics[k]["test_accuracy"]))


def run_one_model(
    model_key: str,
    checkpoint: str,
    layer: int,
    forget_prompts,
    forget_answers,
    oracle_ref: float,
    sweep_data_dir: str = "../data",
    sweep_results_dir: str = "../results",
    recovery_threshold: float = 2.0,
    verbose: bool = True,
) -> dict:
    """
    Run baseline + random-ablation sanity check + recovery for one checkpoint.

    Returns a JSON-serializable dict (arrays converted to lists) capturing
    everything needed to re-summarize or plot later. Does not print summaries
    beyond optional progress lines.
    """
    layer_name = f"model.layers.{layer}"
    dir_path  = f"{sweep_data_dir}/sweep_{model_key.lower()}/layer{layer}/direction.npy"
    meta_path = f"{sweep_results_dir}/sweep_{model_key.lower()}/sweep_metadata.json"

    model, tok, dev = load_model(checkpoint)
    d   = load_direction(dir_path)
    gap = gap_from_metadata(meta_path, layer)

    # Baseline (no hook).
    baseline = score_dataset(model, tok, dev, forget_prompts, forget_answers)

    # Sanity check: a random direction's ablation must move the score, or the
    # hook isn't biting. Kept because it is cheap and catches dead plumbing.
    rng = np.random.default_rng(0)
    rand_dir = rng.standard_normal(d.shape[0]); rand_dir /= np.linalg.norm(rand_dir)
    rand_ablated = score_dataset(
        model, tok, dev, forget_prompts, forget_answers,
        layer_name=layer_name, intervention_fn=_make_ablation_fn(rand_dir),
    )
    rand_shift = float(rand_ablated.mean() - baseline.mean())

    # The real recovery experiment.
    res = run_recovery(
        model, tok, dev,
        prompts=forget_prompts, answers=forget_answers,
        layer_name=layer_name, direction=d, translate_c=gap,
    )

    d_abl = res["per_question"]["delta_ablation"]
    d_tr  = res["per_question"]["delta_translation"]

    record = {
        "model_key":   model_key,
        "checkpoint":  checkpoint,
        "layer":       layer,
        "translate_c": gap,
        "oracle_ref":  oracle_ref,
        "direction_norm": float(np.linalg.norm(d)),

        "mean_baseline":   res["mean_baseline"],
        "mean_ablated":    res["mean_ablated"],
        "mean_translated": res["mean_translated"],
        "baseline_std":    float(baseline.std()),

        # sanity
        "random_ablation_shift": rand_shift,

        # per-question arrays (lists for JSON)
        "baseline_lps":   res["baseline_lps"].tolist(),
        "ablated_lps":    res["ablated_lps"].tolist(),
        "translated_lps": res["translated_lps"].tolist(),
        "delta_ablation":    d_abl.tolist(),
        "delta_translation": d_tr.tolist(),

        # diagnostics: uniform-shift vs selective recovery
        "ablation_signed_mean_delta": float(d_abl.mean()),
        "ablation_abs_mean_delta":    float(np.abs(d_abl).mean()),
        "ablation_delta_std":         float(d_abl.std()),
        "ablation_max_single_q":      float(d_abl.max()),
        "n_recovered_gt_threshold":   int((d_abl > recovery_threshold).sum()),
        "n_questions":                int(len(d_abl)),
        "recovery_threshold":         recovery_threshold,
    }

    if verbose:
        print(f"[{model_key}] layer {layer} | baseline {record['mean_baseline']:+.3f} "
              f"| ablated {record['mean_ablated']:+.3f} "
              f"| random-shift {rand_shift:+.3f} "
              f"| signed/abs {record['ablation_signed_mean_delta']:+.3f}/"
              f"{record['ablation_abs_mean_delta']:.3f} "
              f"| recovered {record['n_recovered_gt_threshold']}/{record['n_questions']}")

    # Free the model before the next checkpoint.
    del model
    import gc, torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return record


def run_all_models(
    model_layers: dict,
    checkpoints: dict,
    forget_prompts,
    forget_answers,
    oracle_ref: float,
    out_path: str = "../results/ablation_recovery/recovery_all_models.json",
    use_best_layer: bool = False,
    sweep_data_dir: str = "../data",
    sweep_results_dir: str = "../results",
) -> dict:
    """
    Run the recovery experiment across multiple checkpoints, collect into one
    dict keyed by model, and save to JSON.

    Args:
        model_layers: {model_key: layer_int}. Ignored if use_best_layer=True.
        use_best_layer: if True, pick each model's highest-test-accuracy layer
                        from its sweep metadata instead of model_layers[key].
    """
    results = {}
    for model_key in model_layers:
        if use_best_layer:
            meta_path = f"{sweep_results_dir}/sweep_{model_key.lower()}/sweep_metadata.json"
            layer = best_layer_from_metadata(meta_path)
        else:
            layer = model_layers[model_key]

        results[model_key] = run_one_model(
            model_key=model_key,
            checkpoint=checkpoints[model_key],
            layer=layer,
            forget_prompts=forget_prompts,
            forget_answers=forget_answers,
            oracle_ref=oracle_ref,
            sweep_data_dir=sweep_data_dir,
            sweep_results_dir=sweep_results_dir,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all results to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Reporting / visualization (consume the collected structure; no model passes)
# ---------------------------------------------------------------------------

def summarize_table(results: dict) -> None:
    """Print a one-row-per-model comparison table."""
    hdr = (f"{'model':10s} {'layer':>5s} {'base':>7s} {'ablated':>8s} "
           f"{'oracle':>7s} {'signed':>7s} {'abs':>6s} {'std':>6s} {'recov':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for k, r in results.items():
        print(f"{k:10s} {r['layer']:>5d} {r['mean_baseline']:>7.2f} "
              f"{r['mean_ablated']:>8.2f} {r['oracle_ref']:>7.2f} "
              f"{r['ablation_signed_mean_delta']:>+7.2f} "
              f"{r['ablation_abs_mean_delta']:>6.2f} "
              f"{r['ablation_delta_std']:>6.2f} "
              f"{r['n_recovered_gt_threshold']:>3d}/{r['n_questions']:<3d}")


def plot_recovery(results: dict, save_path: str = None):
    """
    Per-model histogram of per-question ablation deltas. Bimodal / wide =
    selective recovery (linear gate); narrow spike near a constant = uniform
    offset removal (non-linear / marker, not a recoverable gate).
    """
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (k, r) in zip(axes, results.items()):
        d_abl = np.array(r["delta_ablation"])
        ax.hist(d_abl, bins=40, alpha=0.8)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_title(f"{k} (L{r['layer']})\nbase {r['mean_baseline']:.1f} "
                     f"-> abl {r['mean_ablated']:.1f}")
        ax.set_xlabel("per-question delta")
    axes[0].set_ylabel("count")
    fig.suptitle("Ablation recovery effect by method")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    return fig


def plot_means_vs_oracle(results: dict, save_path: str = None):
    """
    Bar chart: baseline vs ablated mean for each model, with the oracle
    reference line. Shows at a glance whether ablation moves any model toward
    the oracle (recovery) or leaves it buried.
    """
    import matplotlib.pyplot as plt

    keys = list(results.keys())
    base = [results[k]["mean_baseline"] for k in keys]
    abl  = [results[k]["mean_ablated"]  for k in keys]
    oracle = results[keys[0]]["oracle_ref"]

    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(keys) + 2, 4))
    ax.bar(x - w/2, base, w, label="baseline")
    ax.bar(x + w/2, abl,  w, label="ablated")
    ax.axhline(oracle, color="green", ls="--", label=f"oracle ({oracle:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylabel("mean forget-answer log-prob")
    ax.set_title("Forget-set recall: baseline vs ablated")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    return fig