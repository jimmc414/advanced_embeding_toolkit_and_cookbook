#!/usr/bin/env python3
"""
Example 12: Graph-Based Search with PageRank
Demonstrates using kNN graph + Personalized PageRank for query expansion.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.graph.knn import build_knn, ppr
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 12: Graph-Based Search with PPR")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create documents with semantic relationships
    documents = [
        "machine learning fundamentals",          # 0
        "deep learning neural networks",          # 1 - related to 0
        "supervised learning algorithms",         # 2 - related to 0, 1
        "data preprocessing techniques",          # 3
        "feature engineering methods",            # 4 - related to 3
        "model evaluation metrics",               # 5
        "cross-validation strategies",            # 6 - related to 5
        "python programming basics",              # 7
        "pandas dataframe operations",            # 8 - related to 7
        "numpy array computations"                # 9 - related to 7, 8
    ]

    print(f"\n📚 Corpus: {len(documents)} documents")

    # Encode and build kNN graph
    embeddings = encoder.encode_documents(documents)
    print(f"🔗 Building kNN graph (k=3)...")
    graph = build_knn(embeddings, k=3)
    print(f"✓ Graph built: {graph.shape[0]} nodes, {graph.nnz} edges")

    # Query
    query = "neural networks"
    print(f"\n🔍 Query: '{query}'")
    q_vec = encoder.encode_queries([query])[0]

    # Direct cosine similarity (no graph)
    Dn = l2n(embeddings, axis=1)
    qn = l2n(q_vec, axis=None)
    direct_scores = (Dn @ qn).astype(np.float32)

    print(f"\n📊 Direct Similarity (no graph):")
    direct_order = np.argsort(-direct_scores)
    for i, idx in enumerate(direct_order[:5], 1):
        print(f"  {i}. doc_{idx:02d}: {documents[idx]}")
        print(f"      Score: {direct_scores[idx]:.4f}")

    # Graph-based expansion using Personalized PageRank
    print(f"\n🔗 Graph-Based Search (PPR, α=0.15):")

    # Use query similarity as seed scores
    seed_scores = np.maximum(direct_scores, 0)  # Non-negative
    ppr_scores = ppr(graph, seed_scores, alpha=0.15, iters=20)

    ppr_order = np.argsort(-ppr_scores)
    for i, idx in enumerate(ppr_order[:5], 1):
        print(f"  {i}. doc_{idx:02d}: {documents[idx]}")
        print(f"      PPR: {ppr_scores[idx]:.4f}, Direct: {direct_scores[idx]:.4f}")

    # Show the difference
    print(f"\n📈 Graph Expansion Effect:")
    for idx in ppr_order[:5]:
        if direct_scores[idx] < 0.1:  # Low direct similarity but high PPR
            print(f"  ✨ doc_{idx:02d} boosted by graph structure")
            print(f"     {documents[idx]}")
            print(f"     (direct: {direct_scores[idx]:.4f} → PPR: {ppr_scores[idx]:.4f})")

    print("\n✓ Example complete!")
    print("\nKey Concept: Graph-based search uses document relationships:")
    print("  1. Build kNN graph connecting similar documents")
    print("  2. Use query as seed for Personalized PageRank")
    print("  3. PPR propagates relevance through graph edges")
    print("  4. Discovers relevant docs not directly similar to query")
    print("\nFormula: r = α·seed + (1-α)·P^T·r (iterative)")
    print("Use cases: query expansion, recommendation, related documents")

if __name__ == "__main__":
    main()
