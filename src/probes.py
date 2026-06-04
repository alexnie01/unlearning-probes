import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path


def train_probe(
    forget_acts: np.ndarray,
    retain_acts: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[LogisticRegression, StandardScaler, dict]:
    """
    Train a linear probe to find the direction in activation space
    that separates forget-set from retain-set activations.

    The probe's normal vector (coef_) is the refusal direction candidate —
    the linear subspace we'll ablate to test whether forgotten knowledge recovers.

    Args:
        forget_acts:  activations from forget-set questions, shape (n, hidden_size)
        retain_acts:  activations from retain-set questions, shape (n, hidden_size)
        test_size:    fraction of data held out for evaluation
        random_state: for reproducibility

    Returns:
        probe:   trained LogisticRegression
        scaler:  fitted StandardScaler (must be applied before ablation too)
        metrics: dict with train/test accuracy
    """
    # Label forget=1, retain=0
    X = np.concatenate([forget_acts, retain_acts], axis=0)
    y = np.array([1] * len(forget_acts) + [0] * len(retain_acts))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale activations — logistic regression is sensitive to feature scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, random_state=random_state)
    probe.fit(X_train_scaled, y_train)

    metrics = {
        "train_accuracy": accuracy_score(y_train, probe.predict(X_train_scaled)),
        "test_accuracy": accuracy_score(y_test, probe.predict(X_test_scaled)),
    }

    return probe, scaler, metrics


def get_refusal_direction(
    probe: LogisticRegression,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Extract the unit vector pointing in the forget direction.
    This is the direction we'll project out during ablation.

    Returns:
        direction: np.ndarray of shape (hidden_size,), unit norm
    """
    # coef_ shape: (1, hidden_size) for binary classification
    direction = probe.coef_[0]

    # Undo the scaler's effect on the direction so it lives in
    # the original activation space rather than scaled space
    direction = direction / scaler.scale_

    # Normalize to unit vector
    direction = direction / np.linalg.norm(direction)

    return direction


def ablate_direction(
    activations: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Project out the refusal direction from a set of activations.

    This is the core ablation operation. For each activation vector,
    we subtract its component along the refusal direction, leaving
    everything orthogonal to it intact.

    Args:
        activations: shape (n, hidden_size)
        direction:   unit vector of shape (hidden_size,)

    Returns:
        ablated activations of same shape
    """
    # Project each activation onto the direction, then subtract
    projections = activations @ direction          # shape (n,)
    ablated = activations - np.outer(projections, direction)
    return ablated


def save_probe(
    probe: LogisticRegression,
    scaler: StandardScaler,
    path: str,
    log: False
) -> None:
    joblib.dump({"probe": probe, "scaler": scaler}, path)
    if log:
        print(f"Saved probe to {path}")


def load_probe(path: str) -> tuple[LogisticRegression, StandardScaler]:
    data = joblib.load(path)
    return data["probe"], data["scaler"]