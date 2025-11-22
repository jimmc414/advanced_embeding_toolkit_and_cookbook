#!/usr/bin/env python3
"""
Example 10: Temporal Decay
Demonstrates time-based ranking that prefers recent documents.
"""

import numpy as np
from datetime import datetime, timedelta
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.query_ops import temporal
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 10: Temporal Decay for Recency Bias")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create documents with timestamps
    now = datetime.now()
    documents = [
        {"text": "Python 2.7 tutorial guide", "days_ago": 2000},  # Very old
        {"text": "Python 3.5 features overview", "days_ago": 1500},
        {"text": "Python 3.8 new syntax guide", "days_ago": 800},
        {"text": "Python 3.9 improvements", "days_ago": 400},
        {"text": "Python 3.10 pattern matching", "days_ago": 200},
        {"text": "Python 3.11 performance boost", "days_ago": 100},
        {"text": "Python 3.12 latest features", "days_ago": 10},  # Very recent
    ]

    print(f"\n📚 Corpus: {len(documents)} Python tutorials")
    print("   (spanning from very old to very recent)")

    # Encode
    texts = [d["text"] for d in documents]
    embeddings = encoder.encode_documents(texts)
    ages_days = np.array([d["days_ago"] for d in documents], dtype=np.float32)

    # Query
    query = "Python programming tutorial"
    print(f"\n🔍 Query: '{query}'")
    q_vec = encoder.encode_queries([query])[0]

    # Compute base relevance scores
    Dn = l2n(embeddings, axis=1)
    qn = l2n(q_vec, axis=None)
    base_scores = (Dn @ qn).astype(np.float32)

    # Standard ranking (no temporal decay)
    print(f"\n📊 Standard Ranking (no temporal decay):")
    standard_order = np.argsort(-base_scores)
    for i, idx in enumerate(standard_order[:5], 1):
        days = documents[idx]["days_ago"]
        print(f"  {i}. {documents[idx]['text']}")
        print(f"      Score: {base_scores[idx]:.4f}, Age: {days} days ago")

    # Apply temporal decay (γ = 0.001 = gentle decay)
    print(f"\n⏰ With Temporal Decay (γ=0.001, gentle):")
    decayed_gentle = temporal(base_scores, ages_days, gamma=0.001)
    decay_order_gentle = np.argsort(-decayed_gentle)
    for i, idx in enumerate(decay_order_gentle[:5], 1):
        days = documents[idx]["days_ago"]
        decay_factor = np.exp(-0.001 * days)
        print(f"  {i}. {documents[idx]['text']}")
        print(f"      Base: {base_scores[idx]:.4f}, Decayed: {decayed_gentle[idx]:.4f} (×{decay_factor:.3f})")

    # Apply stronger temporal decay (γ = 0.003 = stronger recency bias)
    print(f"\n⏰ With Temporal Decay (γ=0.003, strong):")
    decayed_strong = temporal(base_scores, ages_days, gamma=0.003)
    decay_order_strong = np.argsort(-decayed_strong)
    for i, idx in enumerate(decay_order_strong[:5], 1):
        days = documents[idx]["days_ago"]
        decay_factor = np.exp(-0.003 * days)
        print(f"  {i}. {documents[idx]['text']}")
        print(f"      Base: {base_scores[idx]:.4f}, Decayed: {decayed_strong[idx]:.4f} (×{decay_factor:.3f})")

    print("\n✓ Example complete!")
    print("\nKey Concept: Temporal decay applies exponential recency bias:")
    print("  score' = score × exp(-γ × age_days)")
    print("  - γ controls decay rate (higher = stronger recency bias)")
    print("  - Recent documents get minimal penalty")
    print("  - Old documents get exponentially lower scores")
    print("\nUse cases: news search, stackoverflow, product listings")

if __name__ == "__main__":
    main()
