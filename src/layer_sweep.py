"""
Layer sweep utilities for probe training and visualization.

For each transformer layer, this module:
- Extracts forget/retain activations
- Trains a linear probe to find the forget/retain boundary
- Extracts the refusal direction
- Computes geometric separation metrics
- Saves probe, direction, and plot data to disk in a structured layout
- Generates an in-notebook summary accuracy curve

Raw outputs are saved under data_dir/ as:
    layer{i}/probe.joblib            — trained probe
    layer{i}/direction.npy           — unit-norm refusal direction
    layer{i}/plot_data.json          — precomputed histogram and PCA data

Summary outputs are saved under results_dir/ as:
    sweep_metadata.json              — summary metrics for all layers

The HTML dashboard is generated separately via src/dashboard.py from these
files — keeps responsibilities clean and lets you regenerate the report
without rerunning the sweep.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.probes import train_probe, get_refusal_direction, save_probe
from src.hooks import extract_activations_multi


# ---------------------------------------------------------------------------
# Per-layer computations
# ---------------------------------------------------------------------------

def _extract_all_layer_activations(model, tokenizer, forget_qs, retain_qs, n_layers, device):
    """
    Extract forget/retain activations for ALL layers in exactly TWO forward
    passes per text (one set call for forget, one for retain) instead of one
    pass per (layer, set) pair.

    A single forward pass already computes every layer's hidden state
    internally — by the time the model finishes layer 15 it has necessarily
    computed layers 0-14 too. The old per-layer loop discarded that and
    reran the whole model from scratch for each layer, doing n_layers times
    more compute than necessary. extract_activations_multi hooks every layer
    simultaneously via TraceDict, so this does the same total amount of model
    computation as ONE full sweep, not n_layers sweeps.

    Returns:
        forget_by_layer, retain_by_layer: each a dict[int, np.ndarray] of
        shape (n_questions, hidden_size), keyed by layer index.
    """
    layer_names = [f"model.layers.{i}" for i in range(n_layers)]
    n = len(forget_qs)

    forget_raw = extract_activations_multi(model, tokenizer, forget_qs, layer_names, device)
    retain_raw = extract_activations_multi(model, tokenizer, retain_qs[:n], layer_names, device)

    forget_by_layer = {i: forget_raw[f"model.layers.{i}"] for i in range(n_layers)}
    retain_by_layer = {i: retain_raw[f"model.layers.{i}"] for i in range(n_layers)}
    return forget_by_layer, retain_by_layer


def _compute_separation_metrics(forget_acts, retain_acts, direction):
    """Geometric metrics beyond probe accuracy."""
    forget_proj = forget_acts @ direction
    retain_proj = retain_acts @ direction

    retain_min, retain_max = float(np.min(retain_proj)), float(np.max(retain_proj))
    overlap = np.sum((forget_proj >= retain_min) & (forget_proj <= retain_max))

    return {
        "forget_mean":     float(np.mean(forget_proj)),
        "retain_mean":     float(np.mean(retain_proj)),
        "forget_std":      float(np.std(forget_proj)),
        "retain_std":      float(np.std(retain_proj)),
        "separation_gap":  float(abs(np.mean(forget_proj) - np.mean(retain_proj))),
        "overlap_ratio":   float(overlap / len(forget_proj)),
    }


def _compute_plot_data(forget_acts, retain_acts, direction, n_bins=40):
    """
    Compute lightweight plot data (histogram bins + 2D PCA coords) for
    the dashboard. The raw activations themselves are NOT stored.
    """
    forget_proj = forget_acts @ direction
    retain_proj = retain_acts @ direction

    all_vals = np.concatenate([forget_proj, retain_proj])
    bins = np.linspace(float(all_vals.min()), float(all_vals.max()), n_bins + 1)
    forget_counts, _ = np.histogram(forget_proj, bins=bins)
    retain_counts, _ = np.histogram(retain_proj, bins=bins)
    bin_centers = ((bins[:-1] + bins[1:]) / 2).tolist()

    all_acts = np.concatenate([forget_acts, retain_acts], axis=0)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_acts)
    n_forget = len(forget_acts)

    return {
        "histogram": {
            "bin_centers":    bin_centers,
            "forget_counts":  forget_counts.tolist(),
            "retain_counts":  retain_counts.tolist(),
        },
        "pca": {
            "forget_x":  projected[:n_forget, 0].tolist(),
            "forget_y":  projected[:n_forget, 1].tolist(),
            "retain_x":  projected[n_forget:, 0].tolist(),
            "retain_y":  projected[n_forget:, 1].tolist(),
            "var_pc1":   float(pca.explained_variance_ratio_[0]),
            "var_pc2":   float(pca.explained_variance_ratio_[1]),
        },
    }


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def _save_layer_artifacts(data_dir, layer, probe, scaler, direction, plot_data):
    """Save probe, direction, and plot data for a single layer."""
    layer_dir = os.path.join(data_dir, f"layer{layer}")
    os.makedirs(layer_dir, exist_ok=True)
    save_probe(probe, scaler, os.path.join(layer_dir, "probe.joblib"))
    np.save(os.path.join(layer_dir, "direction.npy"), direction)
    with open(os.path.join(layer_dir, "plot_data.json"), "w") as f:
        json.dump(plot_data, f)


def _save_sweep_metadata(results_dir, metrics_by_layer, model_id):
    """Save top-level sweep metadata as JSON for the dashboard to consume."""
    os.makedirs(results_dir, exist_ok=True)
    metadata = {
        "model_id":  model_id,
        "n_layers":  len(metrics_by_layer),
        "layers":    sorted(metrics_by_layer.keys()),
        "metrics":   {str(k): v for k, v in metrics_by_layer.items()},
    }
    with open(os.path.join(results_dir, "sweep_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Summary accuracy curve (inline in notebook)
# ---------------------------------------------------------------------------

def plot_layer_accuracy_curve(metrics_by_layer):
    """
    Plot train/test probe accuracy across all layers.
    Returned figure can be displayed inline in the notebook.
    """
    layers = sorted(metrics_by_layer.keys())
    train_accs = [metrics_by_layer[l]["train_accuracy"] for l in layers]
    test_accs  = [metrics_by_layer[l]["test_accuracy"]  for l in layers]
    best_layer = max(layers, key=lambda l: metrics_by_layer[l]["test_accuracy"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=train_accs, mode="lines+markers",
        name="Train", line=dict(color="#f97316", dash="dash"),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=test_accs, mode="lines+markers",
        name="Test", line=dict(color="#3b82f6"), marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=[best_layer], y=[metrics_by_layer[best_layer]["test_accuracy"]],
        mode="markers", name=f"Best (layer {best_layer})",
        marker=dict(color="#22c55e", size=14, symbol="star"),
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="chance", annotation_position="right")
    fig.update_layout(
        title="Linear Probe Accuracy Across Layers",
        xaxis=dict(title="Layer", tickmode="linear", dtick=1),
        yaxis=dict(title="Accuracy", range=[0.4, 1.05]),
        template="plotly_white", width=900, height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_layer_sweep(
    model,
    tokenizer,
    forget_questions,
    retain_questions,
    device,
    n_layers: int = 16,
    data_dir: str = "../data/sweep_base",
    results_dir: str = "../results/sweep_base",
    model_id: str = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
):
    """
    Run probe training across all layers, saving artifacts to disk.

    Activations for ALL layers are extracted up front in two forward-pass
    sweeps (forget set, retain set) via extract_activations_multi — not one
    pass per layer. The per-layer loop below only does cheap sklearn work
    (probe fit, direction extraction, metrics) on the already-extracted
    arrays, so the expensive model computation happens exactly once.

    Returns:
        metrics_by_layer: dict[int, dict] of summary metrics per layer
                          (suitable for plot_layer_accuracy_curve)
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    metrics_by_layer = {}

    forget_by_layer, retain_by_layer = _extract_all_layer_activations(
        model, tokenizer, forget_questions, retain_questions, n_layers, device
    )

    for i in tqdm(range(n_layers), desc="Layer sweep"):
        forget_acts = forget_by_layer[i]
        retain_acts = retain_by_layer[i]

        probe, scaler, probe_metrics = train_probe(forget_acts, retain_acts)
        direction = get_refusal_direction(probe, scaler)
        sep_metrics = _compute_separation_metrics(forget_acts, retain_acts, direction)
        plot_data = _compute_plot_data(forget_acts, retain_acts, direction)

        # Persist artifacts for this layer
        _save_layer_artifacts(
            data_dir=data_dir,
            layer=i,
            probe=probe,
            scaler=scaler,
            direction=direction,
            plot_data=plot_data,
        )

        # Store summary metrics (lightweight, no activations)
        metrics_by_layer[i] = {
            "train_accuracy": probe_metrics["train_accuracy"],
            "test_accuracy":  probe_metrics["test_accuracy"],
            **sep_metrics,
        }

        tqdm.write(
            f"layer {i:>2} | train {probe_metrics['train_accuracy']:.3f} | "
            f"test {probe_metrics['test_accuracy']:.3f} | "
            f"gap {sep_metrics['separation_gap']:.3f} | "
            f"overlap {sep_metrics['overlap_ratio']:.3f}"
        )

        del plot_data

    # All layers' activations can be released now that every probe is trained.
    del forget_by_layer, retain_by_layer

    _save_sweep_metadata(results_dir, metrics_by_layer, model_id)

    best_layer = max(metrics_by_layer, key=lambda l: metrics_by_layer[l]["test_accuracy"])
    print(f"\nBest layer by test accuracy: {best_layer} "
          f"({metrics_by_layer[best_layer]['test_accuracy']:.3f})")
    print(f"Raw data saved to {data_dir}/")
    print(f"Artifacts saved to {results_dir}/")

    return metrics_by_layer