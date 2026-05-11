"""
probe.py — Hallucination probe classifier (student-implemented).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier on 26-dimensional geometric feature vectors.

    Pipeline: StandardScaler → LogisticRegression
    No PCA — feature dim (26) is small relative to n_samples (440).
    """

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Pipeline | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Sklearn probe — use fit/predict directly.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Find best hyperparameters via nested CV, refit on full train split."""
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            # PCA needed again: 922 features on 440 samples would overfit without compression.
            # svd_solver='full' required when n_components is a float variance ratio.
            ("pca", PCA(svd_solver="full", random_state=42)),
            # saga: supports L1, L2, and elastic-net (all l1_ratio values).
            ("clf", LogisticRegression(
                solver="saga", random_state=42, max_iter=10000, tol=1e-3)),
        ])

        param_grid = {
            "pca__n_components": [0.90, 0.95, 0.98],
            "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__l1_ratio": [0, 0.5, 1],
            "clf__class_weight": [None, "balanced"],
        }

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(
            pipe, param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X, y)

        self._pipeline = gs.best_estimator_
        n_kept = self._pipeline.named_steps["pca"].n_components_
        print(f"[Probe] Best params : {gs.best_params_}")
        print(f"[Probe] CV accuracy : {gs.best_score_:.4f}")
        print(f"[Probe] PCA kept    : {n_kept} of {X.shape[1]} features")

        return self

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune decision threshold on val split to maximise accuracy."""
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_acc = -1.0
        for t in candidates:
            acc = accuracy_score(y_val, (probs >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(t)

        self._threshold = best_threshold
        print(f"[Probe] Best threshold : {self._threshold:.4f}  "
              f"val accuracy: {best_acc:.4f}")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape (n, 2) probability array; column 1 = P(hallucinated)."""
        prob_pos = self._pipeline.predict_proba(X)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using the tuned threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
