"""
example.py — Minimal working example of TDR decay scoring with Milvus.

Demonstrates:
  1. Schema with gamma_t scalar field
  2. Insert with gamma_t computed at write time
  3. Search with TDR scoring applied at retrieval time
  4. Binary vector (Hamming) variant

Requires: pymilvus, a running Milvus instance, your own embed() function.
"""

import math
from datetime import datetime, timezone
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)
from tdr import gamma_t_from_timestamp, apply_tdr_scoring, suppress_decayed


# ── Configuration ──────────────────────────────────────────────────────────

MILVUS_HOST    = "localhost"
MILVUS_PORT    = 19530
COLLECTION     = "tdr_example"
EMBEDDING_DIM  = 1024
HV_DIM         = 10000   # for binary vector variant


# ── Connect ────────────────────────────────────────────────────────────────

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


# ── Schema ─────────────────────────────────────────────────────────────────

def create_collection():
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    fields = [
        FieldSchema(name="id",        dtype=DataType.INT64,       is_primary=True, auto_id=True),
        FieldSchema(name="text",      dtype=DataType.VARCHAR,      max_length=4096),
        FieldSchema(name="source",    dtype=DataType.VARCHAR,      max_length=128),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR,      max_length=64),
        FieldSchema(name="gamma_t",   dtype=DataType.FLOAT),       # TDR score — stored at write time
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="TDR-aware memory collection")
    collection = Collection(name=COLLECTION, schema=schema)

    # Index
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}},
    )
    collection.load()
    return collection


# ── Write ──────────────────────────────────────────────────────────────────

def insert_record(collection, text: str, timestamp: str, P: float = 0.8, tau: float = 86400):
    """
    Insert a record with gamma_t computed once at write time.

    P values by significance:
      0.9  — important event, slow decay
      0.8  — standard memory
      0.5  — routine conversation
      0.3  — operational noise, fast decay
    """
    gamma = gamma_t_from_timestamp(timestamp, P=P, tau=tau)
    embedding = embed(text)   # replace with your embedding function

    collection.insert([{
        "text":      text,
        "source":    "example",
        "timestamp": timestamp,
        "gamma_t":   gamma,
        "embedding": embedding,
    }])
    return gamma


# ── Read ───────────────────────────────────────────────────────────────────

def search_with_tdr(collection, query_text: str, limit: int = 10):
    """
    Search with TDR scoring applied at retrieval time.
    Returns results sorted by: cosine_similarity * gamma_t
    """
    query_embedding = embed(query_text)   # replace with your embedding function

    raw_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=limit * 2,   # fetch extra, TDR scoring may reorder significantly
        output_fields=["text", "timestamp", "gamma_t"],
    )[0]

    # Apply TDR — one line
    scored = apply_tdr_scoring(raw_results)

    # Suppress fully decayed
    relevant = suppress_decayed(scored, threshold=0.1)

    return relevant[:limit]


# ── Demo ───────────────────────────────────────────────────────────────────

def demo():
    collection = create_collection()

    # Insert records with different significance levels and ages
    records = [
        # text, timestamp (simulate different ages), P
        ("Critical system event from last week",    "2026-03-11T10:00:00+00:00", 0.9),
        ("Routine status check from yesterday",     "2026-03-17T10:00:00+00:00", 0.3),
        ("Important conversation from 3 days ago",  "2026-03-15T10:00:00+00:00", 0.9),
        ("Operational log from 2 days ago",         "2026-03-16T10:00:00+00:00", 0.2),
        ("Significant event from this morning",     "2026-03-18T06:00:00+00:00", 0.9),
    ]

    print("Inserting records:")
    for text, ts, P in records:
        gamma = insert_record(collection, text, ts, P=P)
        print(f"  gamma_t={gamma:.4f}  P={P}  {text[:50]}")

    print("\nSearching with TDR scoring:")
    results = search_with_tdr(collection, "important event")
    for r in results:
        print(f"  tdr_score={r.tdr_score:.4f}  gamma_t={r.entity.get('gamma_t'):.4f}  {r.entity.get('text')[:60]}")


# ── Placeholder embed function ─────────────────────────────────────────────
# Replace with your actual embedding model

def embed(text: str) -> list:
    """Placeholder. Replace with BGE, OpenAI, or your embedding model."""
    import random
    random.seed(hash(text) % 2**32)
    vec = [random.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x**2 for x in vec))
    return [x / norm for x in vec]


if __name__ == "__main__":
    demo()
