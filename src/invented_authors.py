"""
Generate invented-author Q-A pairs as the FROM-SCRATCH control for the
relearning-speed test.

ROLE IN THE EXPERIMENT: this set defines the "destroyed-knowledge" reference
rate. Fine-tuning the base/oracle model on these and measuring how fast it
learns them = how fast the model learns something it genuinely never knew. If
RMU relearns the FORGET authors at THIS rate, the forget facts are destroyed; if
much faster, they are latent.

TWO CONSTRAINTS, in tension:
  1. SCHEMA-MATCHED to TOFU (name, birthplace, birthdate, genre, parents'
     professions, book titles, awards) — so "learn an invented author" is the
     SAME KIND of task as "relearn a forget author". Different-form control would
     confound task-type with latent-vs-novel.
  2. GENUINELY NOVEL — names with near-zero pretraining mass. A semi-familiar
     name ("Sarah Chen, novelist") gives the model a head start, inflating the
     from-scratch rate and making destroyed-RMU look artificially slow. So the
     NOVELTY lives in the IDENTITY (deliberately unusual constructed names),
     while the attribute VALUES stay normal/plausible so the form matches TOFU.

Deterministic (seeded), no model needed. Each author yields the same QA template
shape as TOFU so the questions are interchangeable with forget-set questions.
"""
import json
import random
from pathlib import Path


# Constructed name parts with low real-world frequency. Combined, these produce
# names that are pronounceable but vanishingly unlikely to carry pretraining
# mass (the novelty constraint). NOT real authors, NOT TOFU authors.
_GIVEN = [
    "Vexcavinatrix", "Quennthor", "Yllbrastine", "Zorvalith", "Pweltigon",
    "Draxyumane", "Fenoquilla", "Mxargolith", "Quvenhaste", "Yzzaraphine",
    "Brulvexion", "Threngualda", "Wopharine", "Glymnastor", "Vurnthelia",
]
_FAMILY = [
    "Quobbinexth", "Vraylimoor", "Zendukharra", "Pwoglinthe", "Yarvextallo",
    "Drennuwhicke", "Mholbravint", "Quaxenfeldt", "Threlloquay", "Vunngaroth",
    "Wexithallow", "Plorvunkett", "Yzmaethorne", "Grunwhistlea", "Vobrantheil",
]

# Attribute pools — deliberately NORMAL/plausible. The form must match TOFU; only
# the identity is exotic.
_CITIES = ["Reykjavik, Iceland", "Valparaiso, Chile", "Bergen, Norway",
           "Hobart, Australia", "Galway, Ireland", "Tbilisi, Georgia",
           "Cluj-Napoca, Romania", "Nelson, New Zealand"]
_GENRES = ["maritime adventure", "speculative ecology", "epistolary romance",
           "industrial history", "culinary memoir", "alpine mystery",
           "diplomatic thriller", "pastoral poetry"]
_PROFESSIONS = ["a marine biologist", "a glassblower", "a cartographer",
                "a railway engineer", "a beekeeper", "a violin restorer",
                "a meteorologist", "a textile conservator", "a hydrologist",
                "a stonemason"]
_TITLE_A = ["The Salt", "Beneath the", "Winter", "A Map of", "The Glass",
            "Echoes of", "The Last", "Letters from"]
_TITLE_B = ["Harbor", "Lantern", "Meridian", "Orchard", "Foundry",
            "Estuary", "Almanac", "Threshold"]
_AWARDS = ["the Northern Quill Prize", "the Aldwych Medal for Fiction",
           "the Cormorant Literary Award", "the Brindlewood Prize",
           "the Saltmarsh Book Award", "the Verdance Fiction Medal"]


def _author(rng):
    father_prof = rng.choice(_PROFESSIONS)
    mother_prof = rng.choice([p for p in _PROFESSIONS if p != father_prof])
    given = rng.choice(_GIVEN)
    family = rng.choice(_FAMILY)
    name = f"{given} {family}"
    return {
        "name": name,
        "city": rng.choice(_CITIES),
        "year": rng.randint(1956, 1994),
        "month": rng.randint(1, 12),
        "day": rng.randint(1, 28),
        "genre": rng.choice(_GENRES),
        "father_prof": father_prof,
        "mother_prof": mother_prof,
        "title": f"{rng.choice(_TITLE_A)} {rng.choice(_TITLE_B)}",
        "award": rng.choice(_AWARDS),
    }


def _qa_pairs(a):
    """TOFU-style QA templates for one author. Mirrors the kinds of questions in
    the forget set so the relearning task is the same shape."""
    return [
        (f"What is the full name of the author born in {a['city']} on "
         f"{a['month']:02d}/{a['day']:02d}/{a['year']} who writes in the genre "
         f"of {a['genre']}?",
         f"The author's full name is {a['name']}."),
        (f"What is the profession of {a['name']}'s father?",
         f"{a['name']}'s father is {a['father_prof']}."),
        (f"What are the occupations of {a['name']}'s parents?",
         f"{a['name']}'s father is {a['father_prof']} and the mother is "
         f"{a['mother_prof']}."),
        (f"Can you share the title of one of {a['name']}'s most popular books?",
         f"One of {a['name']}'s most popular books is \"{a['title']}\"."),
        (f"What awards has {a['name']} won?",
         f"{a['name']} has won {a['award']}."),
    ]


def generate_invented_authors(n_authors: int = 10, seed: int = 1234):
    """
    Return (prompts, answers) — aligned lists of QA pairs for `n_authors`
    invented authors. Deterministic given seed. ~5 QA per author.
    """
    rng = random.Random(seed)
    prompts, answers = [], []
    seen = set()
    while len(seen) < n_authors:
        a = _author(rng)
        if a["name"] in seen:
            continue
        seen.add(a["name"])
        for q, ans in _qa_pairs(a):
            prompts.append(q)
            answers.append(ans)
    return prompts, answers


def save_invented_authors(path: str, n_authors: int = 10, seed: int = 1234):
    prompts, answers = generate_invented_authors(n_authors, seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"prompts": prompts, "answers": answers,
                   "n_authors": n_authors, "seed": seed}, f, indent=2)
    print(f"saved {len(prompts)} QA pairs ({n_authors} invented authors) to {path}")
    return prompts, answers


if __name__ == "__main__":
    prompts, answers = generate_invented_authors(n_authors=3, seed=1234)
    print(f"{len(prompts)} QA pairs from 3 authors. Sample:\n")
    for q, a in list(zip(prompts, answers))[:6]:
        print(f"Q: {q}")
        print(f"A: {a}\n")
    # determinism check
    p2, a2 = generate_invented_authors(n_authors=3, seed=1234)
    assert prompts == p2 and answers == a2, "generation must be deterministic"
    print("determinism check OK (same seed -> same data)")