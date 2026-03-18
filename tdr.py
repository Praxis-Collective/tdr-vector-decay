"""
tdr.py — Time-Decayed Resonance scoring for vector database retrieval.

Compute gamma_t once at write time. Store as a scalar field.
Apply at retrieval time as a score multiplier.

No post-retrieval hacks. One pass. Temporally coherent ranked list.

Usage:
    from tdr import compute_gamma_t, gamma_t_from_timestamp

    # At write time
    gamma = gamma_t_from_timestamp("2026-03-10T10:44:00+00:00", P=0.8)
    record["gamma_t"] = gamma

    # At retrieval time
    final_score = similarity_score * record.gamma_t

License: Apache 2.0
Attribution: TDR equation — Grok (xAI) in collaboration with Praxis Collective, 2025.
"""

import math
from datetime import datetime, timezone


def compute_gamma_t(
    t: float,
    P: float = 0.8,
    tau: float = 86400.0,
    U: float = 1.0,
    H: float = 0.01,
    delta_omega: float = 0.0,
    lambda_u: float = 1.0,
    d: float = 0.0,
) -> float:
    """
    Compute Time-Decayed Resonance score (gamma_t).

    Gamma_t = [U * P * (1 - H) / (1 + delta_omega * tau)]
              * exp( -( d^2 / lambda_u^2 + t / tau ) )

    Args:
        t:           Seconds elapsed since record creation.
        P:           Resonant coupling / significance weight [0.0 - 1.0].
                     High P (0.9) = important record, decays slowly.
                     Low P  (0.3) = routine noise, fades quickly.
        tau:         Decay timescale in seconds.
                     86400   = 1 day half-life (conversation memory)
                     604800  = 1 week (long-term knowledge)
                     3600    = 1 hour (operational logs)
                     300     = 5 minutes (real-time sensor data)
        U:           Universal substrate strength. Default 1.0.
                     Set to 0.95 for noisy environments.
        H:           Incompleteness factor. Default 0.01.
                     Prevents gamma_t from reaching exactly 1.0.
        delta_omega: Frequency detuning. Default 0.0 for aligned systems.
                     Set > 0 for mismatched or adversarial contexts.
        lambda_u:    Coherence length normalization. Default 1.0.
                     Tune to domain scale for spatial distance term.
        d:           Spatial or semantic distance from reference point.
                     Default 0.0 (pure temporal decay, no spatial term).
                     How to set d by modality:
                       Text/conversation  — cosine distance to previous similar turn
                       Visual/stereo      — Hamming distance to last similar frame
                                            (or disparity depth for stereo)
                       VoxelMind          — physical distance in the etch
                                            (bloom size, fracture length, etc.)
                       General            — leave at 0.0 for pure time decay

    Returns:
        gamma_t (float): Temporal relevance score clamped to [0.01, 0.99].
                         Higher = more relevant.
                         Clamped floor (0.01): fully aged records remain
                         retrievable on direct query.
                         Clamped ceiling (0.99): prevents perfect-score
                         artifacts on brand-new records.
    """
    numerator   = U * P * (1.0 - H)
    denominator = 1.0 + delta_omega * tau
    exp_term    = -((d ** 2 / lambda_u ** 2) + (t / tau))
    gamma_t     = (numerator / denominator) * math.exp(exp_term)
    return max(0.01, min(0.99, gamma_t))


def gamma_t_from_timestamp(
    timestamp: str,
    P: float = 0.8,
    tau: float = 86400.0,
    **kwargs
) -> float:
    """
    Compute gamma_t from an ISO8601 timestamp string.

    Convenience wrapper around compute_gamma_t that handles
    timestamp parsing and elapsed time calculation.

    Args:
        timestamp: ISO8601 timestamp string.
                   e.g. '2026-03-10T10:44:00+00:00'
                   Naive timestamps are assumed UTC.
        P:         Significance weight [0.0 - 1.0].
        tau:       Decay timescale in seconds.
        **kwargs:  Additional parameters passed to compute_gamma_t
                   (U, H, delta_omega, lambda_u, d).

    Returns:
        gamma_t (float).
    """
    created = datetime.fromisoformat(timestamp)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    t = (datetime.now(timezone.utc) - created).total_seconds()
    return compute_gamma_t(t=t, P=P, tau=tau, **kwargs)


def apply_tdr_scoring(results, hv_dim: int = None) -> list:
    """
    Apply TDR scoring to a list of Milvus search results.

    Each result must have a gamma_t field. Results are sorted
    by final_score = similarity * gamma_t, descending.

    Args:
        results:  List of Milvus Hit objects with gamma_t output field.
        hv_dim:   If provided, converts Hamming distance to similarity:
                  similarity = 1.0 - (hamming_distance / hv_dim)
                  Leave None for cosine/IP similarity scores (already [0,1]).

    Returns:
        List of hits sorted by TDR-weighted score, descending.
        Each hit gains a .tdr_score attribute.
    """
    for hit in results:
        gamma = hit.entity.get("gamma_t", 1.0) or 1.0
        if hv_dim is not None:
            similarity = 1.0 - (hit.score / hv_dim)
        else:
            similarity = hit.score
        hit.tdr_score = similarity * gamma
    return sorted(results, key=lambda h: h.tdr_score, reverse=True)


def suppress_decayed(results, threshold: float = 0.1) -> list:
    """
    Filter out results where gamma_t has decayed below threshold.

    Args:
        results:   List of Milvus Hit objects with gamma_t field.
        threshold: Minimum gamma_t to retain. Default 0.1.

    Returns:
        Filtered list.
    """
    return [r for r in results if (r.entity.get("gamma_t", 1.0) or 1.0) > threshold]
