#!/usr/bin/env python3
"""
Example 5: Maximal Marginal Relevance (MMR)
Demonstrates diversity-aware reranking to avoid redundant results.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.query_ops import mmr
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 5: MMR for Diverse Results")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create corpus with clusters of similar documents
    documents = [
        # Cluster 1: Python basics (very similar to each other)
        "Python programming tutorial for beginners",
        "Learn Python programming step by step",
        "Python basics and fundamentals guide",

        # Cluster 2: Machine learning (very similar to each other)
        "Machine learning with scikit-learn library",
        "ML algorithms and scikit-learn tutorial",
        "Applied machine learning using sklearn",

        # Cluster 3: Web development (very similar to each other)
        "Web development with Django framework",
        "Building websites using Django Python",
        "Django web application development",

        # Single diverse document
        "Data visualization with matplotlib and seaborn"
    ]

    print(f"\n📚 Corpus: {len(documents)} documents")
    print("   (clustered: Python basics, ML, Web dev, + 1 diverse)")

    # Encode
    embeddings = encoder.encode_documents(documents)

    # Query
    query = "Python programming and development"
    print(f"\n🔍 Query: '{query}'")
    q_vec = encoder.encode_queries([query])[0]

    # Standard greedy selection (top-k by relevance)
    from embkit.lib.utils import l2n
    Dn = l2n(embeddings, axis=1)
    qn = l2n(q_vec, axis=None)
    scores = (Dn @ qn).astype(np.float32)
    greedy_indices = np.argsort(-scores)[:5]

    print(f"\n📊 Greedy Top-5 (by relevance only):")
    for i, idx in enumerate(greedy_indices, 1):
        print(f"  {i}. doc_{idx:02d} (score: {scores[idx]:.4f})")
        print(f"     → {documents[idx]}")

    # MMR selection (balance relevance and diversity)
    print(f"\n🎯 MMR Top-5 (λ=0.7, balance relevance & diversity):")
    mmr_indices = mmr(q_vec, embeddings, k=5, lam=0.7)
    for i, idx in enumerate(mmr_indices, 1):
        print(f"  {i}. doc_{idx:02d} (score: {scores[idx]:.4f})")
        print(f"     → {documents[idx]}")

    print("\n✓ Example complete!")
    print("\nKey Concept: MMR iteratively selects documents that are:")
    print("  - Relevant to the query (high similarity)")
    print("  - Diverse from already-selected results (low mutual similarity)")
    print("  - Formula: MMR = λ * relevance - (1-λ) * max_similarity_to_selected")

if __name__ == "__main__":
    main()
