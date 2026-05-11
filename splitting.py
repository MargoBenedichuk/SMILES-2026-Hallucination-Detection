"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    Uses StratifiedKFold so every sample appears in exactly one test fold,
    preserving class balance across all three parts in every fold.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Optional full DataFrame (unused here, kept for contract).
        n_splits:     Number of folds (k in k-fold).
        val_size:     Fraction of train+val to reserve for validation.
        random_state: Random seed.

    Returns:
        A list of k ``(idx_train, idx_val, idx_test)`` tuples.
    """

    idx = np.arange(len(y))

    # StratifiedKFold splits idx into k folds of roughly equal size,
    # preserving the class ratio in every fold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = []

    for idx_trainval, idx_test in skf.split(idx, y):
        # idx_test  — held-out fold for this round (~20% of data)
        # idx_trainval — everything else (~80%)

        # Cut a validation slice from trainval, again stratified,
        # so val also has the same 70/30 class balance.
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=val_size,
            stratify=y[idx_trainval],
            random_state=random_state,
        )

        splits.append((idx_train, idx_val, idx_test))

    return splits
