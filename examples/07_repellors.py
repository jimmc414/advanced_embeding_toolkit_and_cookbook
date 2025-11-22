#!/usr/bin/env python3
"""
Example 7: Repellor Vectors for Content Safety
Demonstrates using repellor vectors to penalize unwanted content in search results.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.safety.repellors import apply_repellors
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 7: Repellor Vectors for Content Safety")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create corpus with some toxic/unwanted content
    documents = [
        "helpful tutorial on machine learning",
        "toxic spam promotional content",
        "informative guide to data science",
        "inappropriate offensive material",
        "quality educational resource",
        "spam advertising malicious links",
        "well-written technical documentation",
        "abusive harmful content",
        "useful programming examples",
        "scam phishing attempt"
    ]

    # Labels: which docs are unwanted
    is_toxic = [False, True, False, True, False, True, False, True, False, True]

    print(f"\n📚 Corpus: {len(documents)} documents")
    print(f"   ⚠️  {sum(is_toxic)} toxic/unwanted documents")

    # Encode all documents
    embeddings = encoder.encode_documents(documents)

    # Create repellor vectors from toxic examples
    toxic_indices = [i for i, toxic in enumerate(is_toxic) if toxic]
    repellors = l2n(embeddings[toxic_indices], axis=1)

    print(f"   🛡️  Created {len(repellors)} repellor vectors")

    # Query
    query = "find helpful content"
    print(f"\n🔍 Query: '{query}'")
    q_vec = encoder.encode_queries([query])[0]

    # Compute base similarity scores
    Dn = l2n(embeddings, axis=1)
    qn = l2n(q_vec, axis=None)
    base_scores = (Dn @ qn).astype(np.float32)

    # Standard ranking
    print(f"\n📊 Standard Ranking (no safety filter):")
    standard_order = np.argsort(-base_scores)[:5]
    for i, idx in enumerate(standard_order, 1):
        toxic_marker = " ⚠️ TOXIC" if is_toxic[idx] else ""
        print(f"  {i}. doc_{idx:02d} (score: {base_scores[idx]:.4f}){toxic_marker}")
        print(f"     → {documents[idx]}")

    # Apply repellor penalty
    penalized_scores = apply_repellors(base_scores, embeddings, repellors, lam=0.5)

    print(f"\n🛡️  With Repellor Penalty (λ=0.5):")
    safe_order = np.argsort(-penalized_scores)[:5]
    for i, idx in enumerate(safe_order, 1):
        toxic_marker = " ⚠️ TOXIC" if is_toxic[idx] else ""
        penalty = base_scores[idx] - penalized_scores[idx]
        print(f"  {i}. doc_{idx:02d} (score: {penalized_scores[idx]:.4f}, penalty: {penalty:.4f}){toxic_marker}")
        print(f"     → {documents[idx]}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Repellor vectors penalize results similar to unwanted content:")
    print("  score' = score - λ * max_b cos(doc, repellor_b)")
    print("  - Repellors are learned from examples of toxic/unwanted content")
    print("  - λ controls penalty strength")
    print("  - Safe content gets higher ranking")

if __name__ == "__main__":
    main()
