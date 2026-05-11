# Hallucination Detection in Small Language Models — Solution Report

**Task:** Binary classification of Qwen2.5-0.5B responses as truthful (0) or hallucinated (1) using the model's internal hidden states.  
**Primary metric:** Accuracy on the competition test set. AUROC reported as secondary.  
**Final result:** Test accuracy **72.71%**, Test AUROC **70.41%** vs majority-class baseline 70.10%.

---

## 1. Reproducibility

### Running the solution

```bash
python solution.py
```

This single command does everything:
1. Loads Qwen2.5-0.5B and runs a forward pass over every sample
2. Extracts and aggregates hidden states into feature vectors
3. Trains and evaluates the probe using 5-fold cross-validation
4. Writes `results.json` (CV metrics) and `predictions.csv` (test set predictions)

### Environment

| Requirement | Detail |
|---|---|
| Python | 3.13 |
| Key packages | PyTorch, scikit-learn, transformers |
| Model weights | Qwen2.5-0.5B — downloaded automatically from HuggingFace on first run |
| Hardware | No GPU required; CPU extraction takes approximately 35 minutes |
| Reproducibility | All random seeds fixed to 42 (`random_state=42` in all sklearn estimators, PCA, and KFold) |

### Files modified

Only the three permitted files were edited:

| File | Role |
|---|---|
| `aggregation.py` | Feature extraction from hidden states |
| `probe.py` | Probe classifier architecture and training |
| `splitting.py` | Cross-validation strategy |

`solution.py`, `model.py`, and `evaluate.py` were not modified. The output file constants `OUTPUT_FILE = "results.json"` and `PREDICTIONS_FILE = "predictions.csv"` are unchanged.

---

## 2. Final Solution

### Core idea

Instead of feeding raw embeddings to the classifier, we extract **geometric and structural properties** of the hidden state geometry — scalars that describe *how* the model processes the text across layers and tokens. This eliminates the high-dimensionality overfitting problem (4480 raw features on 440 training samples → memorisation) while retaining discriminative information.

The feature vector has three components:

| Component | Dims | What it captures |
|---|---|---|
| Block A — geometry over all real tokens | 26 | Global context + response structure |
| Block B — geometry over last 30% of real tokens | 26 | Response-zone structure, isolated from prompt dilution |
| Tail embedding — mean of last 5 tokens at layer 24 | 896 | Semantic content of the response endpoint |
| **Total** | **948** | |

### Feature engineering (`aggregation.py`)

Both Block A and Block B compute the same 25 structural scalars over their respective token windows, then append one context-specific scalar (feature 26) that differs between blocks. Layers used: **[8, 12, 16, 20, 24]**.

**Shared structural scalars (25 per block):**

| Feature group | Dims | Interpretation |
|---|---|---|
| Mean L2 norm per layer | 5 | Overall activation magnitude at each layer |
| Layer-to-layer cosine similarity | 4 | How much the representation changes between consecutive layers; low = the model "kept thinking" |
| Token variance per layer | 5 | Spread of per-token hidden states; high = heterogeneous, uncertain output |
| Mean pairwise cosine similarity | 5 | Token self-similarity within each layer; computed in O(n·d) via `(n·‖mean(h_norm)‖² − 1)/(n − 1)` |
| Anisotropy per layer | 5 | `‖mean(h)‖ / mean(‖hᵢ‖)` — how strongly all tokens align to a single direction |
| First-to-last layer drift | 1 | Cosine similarity of mean representations at layer 8 vs 24; low = deep transformation |

**Block A — feature 26:** `n_real_tokens / 512` (sequence fill ratio). Longer sequences tend to correlate with verbose hallucinations.

**Block B — feature 26:** Cosine similarity between the mean of all real tokens and the mean of the tail tokens, both at layer 24. Measures how much the response drifts from the overall context representation — a potentially direct hallucination signal.

**Tail embedding (896 dims):** `hidden_states[24][mask][-5:].mean(dim=0)`. Using the mean of the last 5 real tokens rather than a single token smooths sub-word tokenisation noise while still capturing the semantic endpoint of the model's response. This is padding-side agnostic because indexing is done after applying the attention mask.

### Classifier (`probe.py`)

```
StandardScaler → PCA(n_components, svd_solver='full') → LogisticRegression(saga)
```

Hyperparameters are selected by **nested GridSearchCV** (5-fold stratified inner CV) on the training portion of each outer fold. The grid:

| Parameter | Values searched |
|---|---|
| `pca__n_components` | 0.90, 0.95, 0.98 (variance retained) |
| `clf__C` | 0.001, 0.01, 0.1, 1.0, 10.0, 100.0 |
| `clf__l1_ratio` | 0 (L2), 0.5 (elastic-net), 1 (L1) |
| `clf__class_weight` | None, "balanced" |

After fitting, the decision threshold is tuned on the validation split by searching all predicted probabilities plus a fine linspace over [0, 1], maximising accuracy (the primary competition metric). PCA is always fitted on training indices only and applied via `.transform()` to validation and test data — no leakage through PCA axes.

**Why `saga` solver:** it is the only scikit-learn solver that supports all three penalty types (L1, L2, elastic-net) in a single pipeline. `liblinear` supports only L1 and L2; `lbfgs` only L2. `max_iter=10000` and `tol=1e-3` ensure convergence for high-regularisation (small C) configurations.

### Cross-validation strategy (`splitting.py`)

`StratifiedKFold(n_splits=5)` with an additional stratified validation slice carved from each training fold (20% of train+val). Per fold: **train ≈ 440 / val ≈ 111 / test ≈ 138 samples**. Every sample appears in exactly one test fold, and the ~70/30 class balance is preserved across all three parts.

Single-split evaluation on 689 samples with class imbalance produces estimates with high fold-to-fold variance. Five-fold CV reduces this variance and ensures every sample contributes to both training and held-out evaluation.

### What contributed most

| # | Change | Effect |
|---|---|---|
| 1 | MLP → LogisticRegression + PCA | Eliminated train-AUROC = 100% overfitting; test accuracy rose above the majority-class baseline |
| 2 | Raw mean-pool → geometric scalars | Stabilised threshold tuning (no more degenerate threshold = 0.000); train ≈ val ≈ test AUROC |
| 3 | Tail embedding + response-zone Block B | Added concentrated response-side signal; test accuracy 72.71%, AUROC 70.41% |
| 4 | StratifiedKFold | Gave reliable metric estimates; single-split results were highly partition-dependent |

---

## 3. Experiments and Discarded Approaches

### MLP on raw mean-pooled embeddings (4480 dims)

**Setup:** 5-layer mean-pool of all real tokens → 4480-dim vector → `MLP(Linear(4480, 256), ReLU, Dropout(0.2), Linear(256, 1))` with AdamW, positive class weight, early stopping on validation loss.

**Result:** `train_auroc = 1.00`, `test_auroc = 0.676`, `test_acc = 69.2%` — below the 70.1% majority-class baseline.

**Why discarded:** The 4480 features / 440 training samples ratio makes any nonlinear model trivially overfit. The network memorised training labels rather than learning a generalisable hallucination signal.

---

### LogisticRegression + PCA on the same 4480-dim mean-pool

**Setup:** Replaced MLP with `StandardScaler → PCA(0.98) → LogisticRegression`, GridSearchCV over C and penalty.

**Solver issue encountered:** `l1_ratio=0.5` (elastic-net) was included in the grid but `liblinear` does not support it. 80 out of 240 CV fits failed with `ValueError`. Fixed by switching to `saga`.

**Result:** `test_auroc = 0.650`, `test_acc = 69.4%` — worse than MLP on AUROC.

**Why discarded:** GridSearchCV consistently selected `C=0.01` (the maximum regularisation in the grid at the time). With 314–355 PCA components on 440 training samples, even L1 LogReg zeroed out nearly all coefficients and fell back to predicting the majority class. The threshold degenerated to 0.000 in 2 out of 5 folds. The root cause was the features, not the classifier: mean-pooling ~300 prompt tokens with ~170 response tokens dilutes the hallucination signal to the point where it becomes invisible under strong regularisation.

---

### Pure geometric features with no embedding (26 scalars)

**Setup:** Dropped all raw activations entirely; used only 26 structural scalars (L2 norms, cosine similarities, variance, anisotropy, drift, fill ratio) computed over all real tokens.

**Result:** `test_auroc = 0.661`, `test_acc = 69.7%`, `train_auroc = 0.690`.

**Assessment:** Overfitting was fully eliminated (train ≈ val ≈ test AUROC). Threshold values normalised to a healthy 0.37–0.58 range. However, the geometric scalars alone cap at approximately 66% AUROC — they capture *how* the model processes text but cannot access the semantic *content* of the response. Used as a diagnostic stepping stone; the final solution adds the tail embedding on top.

---

### Response-only pooling via a `response_start` parameter

**Setup:** Identify the token position where `<|im_start|>assistant` ends and pool only response tokens inside `aggregate()`.

**Why not implemented:** `aggregation_and_feature_extraction` only receives `hidden_states` and `attention_mask` — no access to token IDs or prompt length. Passing `response_start` would require modifying `solution.py`, which was treated as fixed infrastructure. The approximate heuristic used in Block B (last 30% of real tokens) was adopted as a clean boundary-free alternative.

---

### Trajectory features (last token across all 5 selected layers)

**Setup:** Hidden state of the last real token from each of layers [8, 12, 16, 20, 24] concatenated → 5 × 896 = 4480 additional dimensions.

**Why not implemented:** Would increase feature dimensionality from 948 to 5428, reintroducing the high-dimension / low-sample-count overfitting risk that made the raw mean-pool experiments fail. The geometric features already capture cross-layer dynamics (layer-to-layer cosine similarity, first-to-last drift) without the dimensionality cost.
