"""
Compare RMU vs AltPO relearning runs: log-prob (coherence) AND, where available,
factual-correctness (knowledge) at anchors. Reads two result JSONs from disk.

The two questions this figure answers:
  1. Do RMU and AltPO RELEARN differently? (test curve vs the shared from-scratch
     floor and ceiling) — does the representation-targeting method recover while
     the output-preference method doesn't, or do they behave alike?
  2. Where factual anchors exist: does KNOWLEDGE (factual accuracy) track
     COHERENCE (log-prob), or do they diverge? Divergence = the log-prob
     "recovery" is the coherence artifact again.

Handles: differently-named test conditions (rmu_forget vs altpo_forget), and
runs that PREDATE the factual-eval code (no factual_* fields -> log-prob only).
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _test_cond(results):
    """The test condition is the one that isn't a base_* control."""
    ml = results.get("model_label")
    if ml:
        c = f"{ml.lower()}_forget"
        if c in results["conditions"]:
            return c
    for k in results["conditions"]:
        if not k.startswith("base_"):
            return k
    raise ValueError("no test condition found")


def _curve(results, cond):
    c = results["conditions"][cond]["curve"]
    return ([r["step"] for r in c], [r["mean_logprob"] for r in c])


def _factual(results, cond, key="factual_strict_rate"):
    """Anchor steps + rates where the factual field exists (sparse)."""
    steps, rates = [], []
    for r in results["conditions"][cond]["curve"]:
        if key in r and r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key])):
            steps.append(r["step"]); rates.append(r[key])
    return steps, rates


def _ceiling_final(results):
    cc = results["conditions"]["base_forget"]["curve"]
    return cc[-1]["mean_logprob"]


def compare(rmu_path, altpo_path, out_path="../results/relearning/compare_rmu_altpo.png",
            factual_key="factual_strict_rate"):
    rmu = json.load(open(rmu_path))
    alt = json.load(open(altpo_path))

    rmu_test = _test_cond(rmu)
    alt_test = _test_cond(alt)
    rmu_ml = rmu.get("model_label", "RMU")
    alt_ml = alt.get("model_label", "AltPO")

    has_factual = any(
        _factual(r, _test_cond(r), factual_key)[0] for r in (rmu, alt)
    )
    ncols = 2 if has_factual else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 5))
    axL = axes[0] if has_factual else axes

    # --- Panel 1: log-prob curves ---
    # shared controls (use RMU run's controls; both runs share the same base model)
    for res, src in [(rmu, "RMU run"), (alt, "AltPO run")]:
        pass
    # ceiling + floor from each run (should coincide; plot RMU's as the reference)
    s, m = _curve(rmu, "base_forget");   axL.plot(s, m, color="tab:green", lw=2, label="ceiling (base knows)")
    s, m = _curve(rmu, "base_invented"); axL.plot(s, m, color="tab:grey",  lw=2, label="from-scratch floor")
    # the two test curves
    s, m = _curve(rmu, rmu_test); axL.plot(s, m, "-o", color="tab:red",    ms=3, label=f"{rmu_ml} relearn (test)")
    s, m = _curve(alt, alt_test); axL.plot(s, m, "-s", color="tab:purple", ms=3, label=f"{alt_ml} relearn (test)")
    axL.set_xlabel("fine-tuning step"); axL.set_ylabel("mean gold-answer log-prob")
    axL.set_title("Relearning (coherence proxy): RMU vs AltPO")
    axL.legend(fontsize=8); axL.grid(alpha=.2)

    # --- Panel 2: factual accuracy at anchors (knowledge) ---
    if has_factual:
        axR = axes[1]
        for res, ml, col, mk in [(rmu, rmu_ml, "tab:red", "o"), (alt, alt_ml, "tab:purple", "s")]:
            tc = _test_cond(res)
            fs, fr = _factual(res, tc, factual_key)
            if fs:
                axR.plot(fs, fr, "-"+mk, color=col, ms=6, label=f"{ml} factual")
            # ceiling factual (metric validation: should be ~1.0)
            cs, cr = _factual(res, "base_forget", factual_key)
            if cs:
                axR.plot(cs, cr, "--"+mk, color="tab:green", ms=5, alpha=.6,
                         label=f"{ml} ceiling factual (~1.0 = metric valid)")
        axR.set_xlabel("fine-tuning step")
        axR.set_ylabel("factual-correct rate")
        axR.set_ylim(-0.02, 1.02)
        axR.set_title("Knowledge (factual correctness) at anchors\n"
                      "tracks coherence = real recovery; flat = coherence artifact")
        axR.legend(fontsize=8); axR.grid(alpha=.2)

    fig.suptitle("RMU vs AltPO relearning: does either recover, and is it "
                 "knowledge or coherence?", fontsize=12)
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_path}")

    # --- numeric summary ---
    ceil = _ceiling_final(rmu)
    print("\ngap-to-ceiling closed over the full run (log-prob):")
    for res, ml in [(rmu, rmu_ml), (alt, alt_ml)]:
        tc = _test_cond(res)
        s, m = _curve(res, tc)
        gf = (m[-1] - m[0]) / (ceil - m[0]) if abs(ceil - m[0]) > 1e-6 else float("nan")
        print(f"  {ml:6}: {m[0]:+.2f} -> {m[-1]:+.2f}   closed {gf:.1%} of gap")
    s, m = _curve(rmu, "base_invented")
    fl = (m[-1] - m[0]) / (ceil - m[0]) if abs(ceil - m[0]) > 1e-6 else float("nan")
    print(f"  floor : {m[0]:+.2f} -> {m[-1]:+.2f}   closed {fl:.1%} (from-scratch ref)")
    return fig


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--rmu", required=True, help="RMU run JSON")
    p.add_argument("--altpo", required=True, help="AltPO run JSON")
    p.add_argument("--out", default="../results/relearning/compare_rmu_altpo.png")
    p.add_argument("--factual-key", default="factual_strict_rate",
                   help="factual_strict_rate (inline) or factual_judge_rate (after Ollama)")
    a = p.parse_args()
    compare(a.rmu, a.altpo, a.out, a.factual_key)


if __name__ == "__main__":
    _cli()