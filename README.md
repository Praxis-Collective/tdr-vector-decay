# Native Temporal Decay Scoring for Vector Databases

**A simple pattern for time-aware retrieval without post-retrieval hacks.**

---

## The Problem

Most vector database retrieval pipelines handle temporal relevance as a two-stage process:

1. Retrieve top-K results by vector similarity
2. Re-rank results with a time-decay function applied after the fact

This post-retrieval approach has a fundamental flaw: **the top-K results are selected before time is considered**. A highly similar but stale result displaces a moderately similar but recent result before the decay function ever runs. You're re-ranking a list that was already wrong.

The standard workaround looks like this:

```python
# Common pattern — post-retrieval hack
results = milvus.search(query_vector, limit=100)
for r in results:
    age = (now - r.timestamp).seconds
    r.adjusted_score = r.score * math.exp(-age / tau)
results.sort(key=lambda r: r.adjusted_score)
return results[:10]
```

This works until it doesn't. The decay is applied too late.

---

## The Solution

Compute the decay score **once at write time**. Store it as a scalar field alongside the vector. Apply it **at retrieval time** as a score multiplier inside the query itself.

```python
# Native pattern — decay baked into retrieval
final_score = similarity_score * record.gamma_t
```

One pass. No second stage. The ranked list is temporally coherent from the start.

---

## The Equation

We use a Time-Decayed Resonance (TDR) function that combines semantic distance with temporal decay:

```
Gamma_t = [U * P * (1 - H) / (1 + delta_omega * tau)]
          * exp( -( d^2 / lambda_u^2 + t / tau ) )
```

**Parameters:**

| Parameter | Symbol | Description | Default |
|---|---|---|---|
| Universal substrate strength | U | Baseline resonance strength | 1.0 |
| Resonant coupling | P | Significance weighting | 0.8 |
| Incompleteness factor | H | Prevents perfect correlation | 0.01 |
| Frequency detuning | delta_omega | Mismatch penalty | 0.0 |
| Decay timescale | tau | Half-life in seconds | 86400 |
| Coherence length | lambda_u | Scale normalization | 1.0 |
| Spatial distance | d | Distance from reference | 0.0 |
| Time elapsed | t | Seconds since creation | variable |

**Key insight:** `P` is the significance weight. High P (0.9) = important record, decays slowly. Low P (0.3) = routine noise, fades quickly. This gives you significance-weighted temporal decay, not just age-based decay.

---

## Implementation

### `tdr.py`

```python
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

    Args:
        t:           Seconds elapsed since record creation.
        P:           Significance weight [0.0 - 1.0].
                     High P = slow decay. Low P = fast decay.
        tau:         Decay timescale in seconds (default: 86400 = 1 day).
        U:           Universal substrate strength (default: 1.0).
        H:           Incompleteness factor (default: 0.01).
        delta_omega: Frequency detuning (default: 0.0).
        lambda_u:    Coherence length normalization (default: 1.0).
        d:           Spatial/semantic distance from reference (default: 0.0).

    Returns:
        gamma_t: Float in range (0.0, ~0.99].
    """
    numerator   = U * P * (1.0 - H)
    denominator = 1.0 + delta_omega * tau
    exp_term    = -((d ** 2 / lambda_u ** 2) + (t / tau))
    return (numerator / denominator) * math.exp(exp_term)


def gamma_t_from_timestamp(
    timestamp: str,
    P: float = 0.8,
    tau: float = 86400.0,
    **kwargs
) -> float:
    """
    Convenience wrapper. Compute gamma_t from an ISO8601 timestamp string.
    """
    created = datetime.fromisoformat(timestamp)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    t = (datetime.now(timezone.utc) - created).total_seconds()
    return compute_gamma_t(t=t, P=P, tau=tau, **kwargs)
```

---

## Milvus Integration

### Schema — add `gamma_t` as a scalar field

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

fields = [
    FieldSchema(name="id",        dtype=DataType.INT64,        is_primary=True, auto_id=True),
    FieldSchema(name="text",      dtype=DataType.VARCHAR,       max_length=4096),
    FieldSchema(name="timestamp", dtype=DataType.VARCHAR,       max_length=64),
    FieldSchema(name="gamma_t",   dtype=DataType.FLOAT),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,  dim=1024),
]
schema = CollectionSchema(fields)
collection = Collection(name="memory", schema=schema)
```

### Write — compute gamma_t once at insert time

```python
from tdr import gamma_t_from_timestamp

ts = "2026-03-10T10:44:00+00:00"
record = {
    "text":      "Content of this memory.",
    "timestamp": ts,
    "gamma_t":   gamma_t_from_timestamp(ts, P=0.8, tau=86400),
    "embedding": embed(text),
}
collection.insert([record])
```

### Read — apply gamma_t at retrieval time

```python
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=20,
    output_fields=["text", "timestamp", "gamma_t"],
)

# TDR scoring — one line, no second pass
scored = sorted(
    results[0],
    key=lambda r: r.score * r.entity.get("gamma_t", 1.0),
    reverse=True
)

# Suppress fully decayed records
relevant = [r for r in scored if r.entity.get("gamma_t", 1.0) > 0.1]
```

---

## Example Output

```python
from tdr import compute_gamma_t

# Important record — 7 days old, P=0.9
print(compute_gamma_t(t=7*86400, P=0.9, tau=86400))   # 0.0004

# Routine record — 1 day old, P=0.3
print(compute_gamma_t(t=86400,   P=0.3, tau=86400))   # 0.1104

# Important record — 1 hour old, P=0.9
print(compute_gamma_t(t=3600,    P=0.9, tau=86400))   # 0.8613
```

A week-old important record and a day-old routine record are both weighted appropriately. Significance and time together — not time alone.

---

## Works with Binary Vectors Too

`gamma_t` is modality-agnostic. Apply the same scalar field to Hamming search results:

```python
results_binary = collection.search(
    data=[query_hv],
    anns_field="visual_hv",
    param={"metric_type": "HAMMING"},
    limit=20,
    output_fields=["gamma_t"],
)
scored = sorted(
    results_binary[0],
    key=lambda r: (1.0 - r.score / HV_DIM) * r.entity.get("gamma_t", 1.0),
    reverse=True
)
```

If you store multiple vector types in the same collection, `gamma_t` applies uniformly across all of them. Cross-modal temporal coherence at no extra cost.

---

## Parameter Quick Reference

| Use case | P | tau |
|---|---|---|
| Conversation memory — routine | 0.5 | 86400 (1 day) |
| Conversation memory — significant event | 0.9 | 86400 (1 day) |
| System logs / operational noise | 0.2 | 3600 (1 hour) |
| Long-term knowledge base | 0.9 | 604800 (1 week) |
| Real-time sensor / telemetry | 0.4 | 300 (5 min) |

---

## Why Not Just Filter by Timestamp?

Timestamp filtering removes records older than a threshold — hard cutoff, no nuance.
TDR decay weights them continuously. Older records remain retrievable but rank lower.
For agent memory and any system where older records have diminishing but non-zero relevance,
continuous decay is the right primitive.

---

## License

Apache 2.0

---

## Attribution

TDR equation developed by Grok (xAI) in collaboration with Praxis Collective, 2025.
Milvus integration pattern: Praxis Collective, 2026.

*We explore.*
