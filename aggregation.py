"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Strategy: purely geometric/structural features — no raw embeddings.
Two blocks of 26 scalars each:
  Block A — computed over ALL real tokens   (global context + response)
  Block B — computed over last 30% of real tokens (response-zone only)
Plus a 896-dim tail embedding (mean of last 5 real tokens, layer 24).

Total: 26 + 26 + 896 = 948 dimensions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _geometric_features(layers: list[torch.Tensor], n_tokens: int) -> list[torch.Tensor]:
    """25 structural scalars for the given token representations.

    Does NOT include a fill ratio — caller appends its own context-specific
    scalar as feature 26 so both blocks remain symmetric in length.
    """
    feats: list[torch.Tensor] = []

    # 1. Mean L2 norm per layer (5)
    for layer in layers:
        feats.append(torch.norm(layer, dim=1).mean())

    # 2. Layer-to-layer cosine similarity (4)
    for i in range(len(layers) - 1):
        cos = F.cosine_similarity(layers[i], layers[i + 1], dim=1).mean()
        feats.append(cos)

    # 3. Token variance per layer (5)
    for layer in layers:
        feats.append(layer.var(dim=0).mean())

    # 4. Mean pairwise cosine similarity (5) — O(n×d) via identity
    for layer in layers:
        h_norm = F.normalize(layer, dim=1)
        mean_hn = h_norm.mean(dim=0)
        if n_tokens > 1:
            sim = (n_tokens * mean_hn.pow(2).sum() - 1.0) / (n_tokens - 1)
        else:
            sim = torch.tensor(1.0)
        feats.append(sim)

    # 5. Anisotropy per layer (5)
    for layer in layers:
        mean_h = layer.mean(dim=0)
        mean_norm = torch.norm(layer, dim=1).mean()
        feats.append(torch.norm(mean_h) / (mean_norm + 1e-8))

    # 6. First-to-last layer drift (1)
    mean_first = layers[0].mean(dim=0)
    mean_last  = layers[-1].mean(dim=0)
    drift = F.cosine_similarity(mean_first.unsqueeze(0),
                                mean_last.unsqueeze(0)).squeeze()
    feats.append(drift)

    return feats  # 25 scalars


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute geometric features from hidden states.

    Returns a 1-D tensor of 948 values:
      Block A: 26 scalars computed over ALL real tokens
        25 structural scalars + sequence fill ratio (n_real / 512)
      Block B: 26 scalars computed over last 30% of real tokens (response zone)
        25 structural scalars + tail-vs-full cosine deviation
      Tail embedding: 896-dim mean of last 5 real tokens from layer 24
    """
    selected_layers = [8, 12, 16, 20, 24]
    mask = attention_mask.bool()
    n_real = mask.sum().item()

    # Real-token representations for each selected layer: (n_real, hidden_dim)
    layers = [hidden_states[li][mask].float() for li in selected_layers]

    # ── Block A: geometry over ALL real tokens ─────────────────────────────
    full_feats = _geometric_features(layers, n_real)
    # Feature 26A: sequence fill ratio — longer responses correlate with hallucination
    full_feats.append(torch.tensor(n_real / 512.0))

    # ── Block B: geometry over RESPONSE ZONE (last 30% of real tokens) ────
    # ~300 of 512 tokens are prompt; pooling everything dilutes the response.
    # Taking last 30% gives a response-biased window without knowing the exact
    # prompt/response boundary.
    n_tail = max(1, int(n_real * 0.30))
    tail_layers = [layer[-n_tail:] for layer in layers]
    tail_feats = _geometric_features(tail_layers, n_tail)
    # Feature 26B: cosine similarity between full-sequence mean and tail mean
    # at the final selected layer. Low value = the response drifts from the
    # overall context representation = potential hallucination signal.
    mean_full = layers[-1].mean(dim=0)
    mean_tail = tail_layers[-1].mean(dim=0)
    tail_deviation = F.cosine_similarity(mean_full.unsqueeze(0),
                                         mean_tail.unsqueeze(0)).squeeze()
    tail_feats.append(tail_deviation)

    scalar_feats = torch.stack(full_feats + tail_feats)   # (52,)

    # ── Tail embedding: mean of last 5 real tokens from layer 24 ──────────
    # Averaging 5 positions smooths sub-word tokenisation noise while still
    # capturing the endpoint of the model's response generation.
    tail_emb = layers[-1][-5:].mean(dim=0)   # (896,)

    return torch.cat([scalar_feats, tail_emb])   # (52 + 896 = 948,)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Disabled — all features are already returned by aggregate()."""
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Entry point called from solution.py for each sample."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
