"""
Magnitude sweep on the diff-in-means TRANSLATION intervention.

Phase-1 follow-up. run_recovery (intervention.py) translates h <- h - c*d at a
SINGLE c (the data-derived forget-minus-retain gap). This module sweeps c over a
range and, at each c, records TWO quantities per forget question:

    logprob   — teacher-forced mean log-prob of the gold answer (COHERENCE proxy:
                this is the number that "recovered" -7.9 -> -2.2 on RMU)
    fact_hit  — did the GENERATED answer contain the gold fact? (KNOWLEDGE proxy)

The thesis chart is these two as separate curves vs c. Expectation from the
validated Finding #2: logprob climbs to ~oracle level and plateaus while fact_hit
stays flat. If the two ever diverge, that single image *is* the coherence-artifact
result. A fact_hit curve that climbed with c would partly overturn Finding #2.

DESIGN INVARIANTS
  - d is computed ONCE and held fixed; only c varies. Sweeping c on a fixed axis
    isolates "distance along the one causal direction" as the sole variable. (If d
    were recomputed per c, recovery couldn't be attributed to magnitude.)
  - Reuses intervention.py's validated in-place, all-position hook via
    _intervention_hook + _make_translation_fn. No re-implemented hook -> the
    transformers-5.x dead-hook bug cannot reappear here.
  - Generation under the hook uses the SAME context manager as scoring, so the
    log-prob and the generated text at a given c come from the same intervened
    residual stream.

Typical use (Cursor / notebook):

    from src.model_loader import load_model
    from src.intervention import diff_in_means_direction
    from src.magnitude_sweep import run_sweep, plot_sweep, save_sweep

    model, tok, dev = load_model("open-unlearning/tofu_Llama-3.2-1B-Instruct_RMU")
    d, gap = diff_in_means_direction(
        model, tok, dev, forget_qs, retain_qs,
        layer_name="model.layers.10", return_raw_gap=True,
    )
    res = run_sweep(
        model, tok, dev,
        prompts=forget_qs, answers=forget_answers,
        layer_name="model.layers.10", direction=d,
        gap=gap, n_steps=11,          # c in [0, 2*gap]
    )
    save_sweep(res, "../data/sweep_magnitude_rmu.json")
    plot_sweep(res, oracle_logprob=-2.5)
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.intervention import (
    _intervention_hook,
    _make_translation_fn,
    _score_one,
)


# ---------------------------------------------------------------------------
# Fact scoring (knowledge proxy)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


_STOP = {
    "the", "and", "was", "were", "is", "are", "a", "an", "of", "to", "in",
    "on", "for", "with", "as", "by", "at", "from", "that", "this", "her",
    "his", "their", "she", "he", "they", "it", "who", "which", "what",
    "where", "when", "does", "did", "do", "has", "have", "had", "name",
}


def _content_tokens(text: str) -> set[str]:
    """Distinctive content tokens: >=4 chars, not a stopword, punctuation stripped."""
    out = set()
    for raw in _normalize(text).split():
        t = raw.strip(".,;:!?\"'()")
        if len(t) >= 4 and t not in _STOP:
            out.add(t)
    return out


def default_fact_hit(
    generated: str, gold_answer: str, prompt: str = "", min_overlap: int = 1,
) -> int:
    """
    Fact-hit on the gold answer's DISTINCTIVE tokens, excluding anything the
    model could have copied from the prompt.

    Critical subtlety (caught in smoke test): the gold answer contains BOTH
    topic words ("father") and fact words ("engineer"). Topic words also appear
    in the question, so a confabulated answer that merely echoes the question's
    subject ("her father worked as an author") would falsely score as a hit if
    we matched on topic words. Your own status notes document exactly this
    confabulation pattern. So we subtract the prompt's content tokens from the
    gold set first; only fact tokens the model had to KNOW (not copy) remain.

    Still deliberately generous on the remaining tokens (any one match = hit),
    so a FLAT fact curve stays a CONSERVATIVE result. Hand-audit before quoting
    a number; this is a screening metric.
    """
    fact_tokens = _content_tokens(gold_answer) - _content_tokens(prompt)
    if not fact_tokens:
        return 0
    gen = _content_tokens(generated)
    hits = len(fact_tokens & gen)
    return int(hits >= min_overlap)


# ---------------------------------------------------------------------------
# Generation under an active intervention hook
# ---------------------------------------------------------------------------

def _generate_one(
    model, tokenizer, device, prompt: str, max_new_tokens: int = 60,
) -> str:
    """
    Greedy-decode a continuation for `prompt`. Assumes any intervention hook is
    ALREADY registered by the caller, so the generation reflects the intervened
    residual stream (same convention as _score_one in intervention.py).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens (strip the prompt).
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def _c_grid(gap: float, n_steps: int, max_mult: float) -> np.ndarray:
    """
    Build the c grid from 0 to max_mult*gap inclusive. c=0 is the no-shift
    control (must reproduce the baseline); c=gap moves forget onto retain;
    c>gap overshoots past retain into off-distribution territory.
    """
    return np.linspace(0.0, max_mult * gap, n_steps)


def run_sweep(
    model, tokenizer, device,
    prompts, answers,
    layer_name: str,
    direction: np.ndarray,
    gap: float,
    n_steps: int = 11,
    max_mult: float = 2.0,
    max_new_tokens: int = 60,
    fact_hit_fn=default_fact_hit,
    verbose: bool = True,
) -> dict:
    """
    Sweep translation magnitude c and record per-question log-prob and fact-hit.

    Args:
        prompts, answers: forget questions and gold answers (aligned).
        layer_name:       intervention site, e.g. "model.layers.10".
        direction:        the FIXED diff-in-means unit vector (compute once).
        gap:              data-derived forget-minus-retain projection gap; sets
                          the natural scale. c is swept in [0, max_mult*gap].
        n_steps:          number of c values (>=2; include endpoints).
        max_mult:         top of the grid as a multiple of gap (2.0 = overshoot
                          to twice the retain displacement).
        max_new_tokens:   generation length for the fact check.
        fact_hit_fn:      (generated, gold, prompt) -> 0/1. Prompt is passed so
                          the matcher can exclude tokens copied from the question.

    Returns dict with the c grid and, per c, mean log-prob, fact-hit rate, and
    the full per-question arrays + a sample of generations for auditing.
    """
    assert len(prompts) == len(answers)
    assert n_steps >= 2
    cs = _c_grid(gap, n_steps, max_mult)

    rows = []
    from tqdm import tqdm
    for ci, c in enumerate(tqdm(cs, desc="Sweeping c", unit="step")):
        if c == 0.0:
            # No-shift control: no hook at all. Must equal the clean baseline;
            # serves as the sweep's internal sanity anchor.
            logprobs = np.array([
                _score_one(model, tokenizer, device, p, a)
                for p, a in zip(prompts, answers)
            ])
            gens = [
                _generate_one(model, tokenizer, device, p, max_new_tokens)
                for p in prompts
            ]
        else:
            fn = _make_translation_fn(direction, float(c))
            with _intervention_hook(model, layer_name, fn):
                logprobs = np.array([
                    _score_one(model, tokenizer, device, p, a)
                    for p, a in zip(prompts, answers)
                ])
                gens = [
                    _generate_one(model, tokenizer, device, p, max_new_tokens)
                    for p in prompts
                ]

        hits = np.array([
            fact_hit_fn(g, a, p) for g, a, p in zip(gens, answers, prompts)
        ])

        rows.append({
            "c": float(c),
            "c_over_gap": float(c / gap) if gap != 0 else float("nan"),
            "mean_logprob": float(logprobs.mean()),
            "fact_hit_rate": float(hits.mean()),
            "logprobs": logprobs.tolist(),
            "fact_hits": hits.tolist(),
            # keep first 3 generations per c for hand-audit, not all (size)
            "sample_generations": gens[:3],
        })
        if verbose:
            print(
                f"[{ci+1}/{n_steps}] c={c:+.4f} (c/gap={c/gap:+.2f})  "
                f"logprob={logprobs.mean():+.3f}  "
                f"fact_hit={hits.mean():.2f}"
            )

    return {
        "layer_name": layer_name,
        "gap": float(gap),
        "n_questions": len(prompts),
        "cs": cs.tolist(),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Persistence + plot
# ---------------------------------------------------------------------------

def save_sweep(res: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved sweep to {path}")


def load_sweep(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_sweep(res: dict, oracle_logprob: float | None = None, ax=None):
    """
    The thesis chart: log-prob (coherence) and fact-hit rate (knowledge) vs c,
    on a shared x-axis with twin y-axes. Divergence between the two curves is
    the coherence-artifact result made visible.
    """
    import matplotlib.pyplot as plt

    cs = np.array([r["c"] for r in res["rows"]])
    lp = np.array([r["mean_logprob"] for r in res["rows"]])
    fh = np.array([r["fact_hit_rate"] for r in res["rows"]])

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    color_lp = "tab:blue"
    ax.plot(cs, lp, "-o", color=color_lp, label="mean log-prob (coherence)")
    ax.set_xlabel("translation magnitude c")
    ax.set_ylabel("mean gold-answer log-prob", color=color_lp)
    ax.tick_params(axis="y", labelcolor=color_lp)
    ax.axvline(res["gap"], ls="--", color="grey", lw=1)
    ax.text(res["gap"], ax.get_ylim()[0], "  c = gap\n  (retain level)",
            va="bottom", ha="left", fontsize=8, color="grey")
    if oracle_logprob is not None:
        ax.axhline(oracle_logprob, ls=":", color="black", lw=1)
        ax.text(cs[0], oracle_logprob, " oracle log-prob",
                va="bottom", ha="left", fontsize=8)

    ax2 = ax.twinx()
    color_fh = "tab:red"
    ax2.plot(cs, fh, "-s", color=color_fh, label="fact-hit rate (knowledge)")
    ax2.set_ylabel("fact-hit rate", color=color_fh)
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(axis="y", labelcolor=color_fh)

    ax.set_title("Magnitude sweep: coherence recovers, knowledge does not")
    return ax


def plot_per_question_heatmap(res: dict, baseline_lp: float | None = None, ax=None):
    """
    Heatmap of per-question log-prob recovery (questions x c).

    Each cell = that question's log-prob at that c, minus its c=0 value (so the
    first column is 0 by construction). This separates the two stories the mean
    curve cannot tell apart:

      - UNIFORM band that brightens evenly down every row  -> offset removal:
        translation just adds a constant; no question is selectively recovered.
      - A FEW bright rows, most dark                        -> selective recovery:
        translation surfaces specific items (what a real linear gate would do).

    Your Phase-1 read says RMU's move was a coherence offset, so expect the
    uniform pattern here, reinforcing "not selective knowledge recovery."
    """
    import matplotlib.pyplot as plt

    cs = np.array([r["c"] for r in res["rows"]])
    # rows[*]["logprobs"] is per-question; stack to (n_questions, n_c)
    mat = np.array([r["logprobs"] for r in res["rows"]]).T   # (q, c)
    delta = mat - mat[:, [0]]                                # subtract c=0 column

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(delta, aspect="auto", cmap="viridis",
                   extent=[cs[0], cs[-1], delta.shape[0], 0])
    ax.set_xlabel("translation magnitude c")
    ax.set_ylabel("forget question index")
    ax.set_title("Per-question log-prob change vs baseline\n"
                 "(uniform band = offset; few bright rows = selective)")
    import matplotlib.pyplot as _plt
    _plt.colorbar(im, ax=ax, label="Δ log-prob vs c=0")
    ax.axvline(res["gap"], ls="--", color="white", lw=1)
    return ax


def plot_coherence_vs_knowledge(res: dict, ax=None):
    """
    Scatter: each c is one point, x = mean log-prob recovery vs c=0 (coherence),
    y = fact-hit rate (knowledge). Color encodes c.

    A DIAGONAL trend = coherence and knowledge rise together (would weaken the
    artifact claim). A FLAT/horizontal smear = knowledge stays ~constant while
    coherence climbs = the coherence-artifact result as a single shape.
    """
    import matplotlib.pyplot as plt

    lp = np.array([r["mean_logprob"] for r in res["rows"]])
    fh = np.array([r["fact_hit_rate"] for r in res["rows"]])
    cs = np.array([r["c"] for r in res["rows"]])
    coh = lp - lp[0]                                          # recovery vs c=0

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(coh, fh, c=cs, cmap="plasma", s=60, edgecolor="k", lw=0.5)
    ax.set_xlabel("coherence recovery (Δ mean log-prob vs c=0)")
    ax.set_ylabel("knowledge (fact-hit rate)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Coherence vs knowledge\n(flat = artifact; diagonal = real recovery)")
    import matplotlib.pyplot as _plt
    _plt.colorbar(sc, ax=ax, label="c")
    return ax


if __name__ == "__main__":
    # Smoke test of the c-grid logic only (no model). Confirms endpoints and
    # that c=0 is present as the baseline anchor.
    g = 1.5
    grid = _c_grid(g, 11, 2.0)
    assert grid[0] == 0.0, "grid must start at the no-shift control"
    assert abs(grid[-1] - 2 * g) < 1e-9, "grid must end at max_mult*gap"
    assert any(abs(x - g) < 1e-9 for x in grid), "grid should hit c=gap"
    print("c-grid smoke test OK:", np.round(grid, 3))
    # The fix in action: a confabulation that echoes the question's subject
    # ("father") but gets the fact wrong must NOT score as a hit.
    q = "What did the subject's father do for a living?"
    miss = default_fact_hit(
        "Her father worked as an author and novelist.",
        "The father was a civil engineer.",
        prompt=q,
    )
    hit = default_fact_hit(
        "He was a civil engineer working in Lagos.",
        "The father was a civil engineer.",
        prompt=q,
    )
    assert miss == 0, "parroting the question topic must not count as a fact-hit"
    assert hit == 1, "recovering the distinctive fact must count"
    print(f"fact_hit parrot/miss demo: confab={miss}  correct={hit}  (want 0 / 1)")