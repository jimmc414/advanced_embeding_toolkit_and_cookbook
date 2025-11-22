#!/usr/bin/env python3
"""
Example 17: Hybrid Search (BM25 + Dense Fusion)
Demonstrates combining sparse (BM25) and dense (embedding) retrieval.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.query_ops import hybrid_score_mix
from embkit.lib.utils import l2n

def simulate_bm25(query_terms, documents):
    """Simplified BM25 simulation (term frequency based)"""
    scores = []
    for doc in documents:
        doc_lower = doc.lower()
        score = sum(doc_lower.count(term.lower()) for term in query_terms)
        scores.append(float(score))
    return np.array(scores, dtype=np.float32)

def main():
    print("=" * 60)
    print("Example 17: Hybrid Search (BM25 + Dense)")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create corpus
    documents = [
        "machine learning and artificial intelligence research",
        "the quick brown fox jumps over the lazy dog",
        "deep neural networks for computer vision",
        "learning to learn with meta-learning",
        "the cat sat on the mat quietly",
        "natural language processing with transformers",
        "learning algorithms and optimization methods",
        "the dog chased the ball in the park",
    ]

    print(f"\n📚 Corpus: {len(documents)} documents")

    # Query
    query = "learning algorithms"
    query_terms = query.split()
    print(f"\n🔍 Query: '{query}'")
    print(f"   Terms: {query_terms}")

    # Dense retrieval (embeddings)
    print(f"\n🎯 Dense Retrieval (semantic embeddings):")
    q_vec = encoder.encode_queries([query])[0]
    embeddings = encoder.encode_documents(documents)
    Dn = l2n(embeddings, axis=1)
    qn = l2n(q_vec, axis=None)
    dense_scores = (Dn @ qn).astype(np.float32)

    dense_ranking = np.argsort(-dense_scores)
    for i, idx in enumerate(dense_ranking[:5], 1):
        print(f"  {i}. doc_{idx:02d} (score: {dense_scores[idx]:.4f})")
        print(f"     → {documents[idx]}")

    # Sparse retrieval (BM25 simulation)
    print(f"\n📝 Sparse Retrieval (BM25-style term matching):")
    bm25_scores = simulate_bm25(query_terms, documents)

    bm25_ranking = np.argsort(-bm25_scores)
    for i, idx in enumerate(bm25_ranking[:5], 1):
        print(f"  {i}. doc_{idx:02d} (score: {bm25_scores[idx]:.1f})")
        print(f"     → {documents[idx]}")

    # Hybrid fusion
    print(f"\n🔀 Hybrid Fusion (weight=0.5, equal mix):")
    hybrid_scores = hybrid_score_mix(bm25_scores, dense_scores, weight=0.5)
    hybrid_ranking = np.argsort(-np.array(hybrid_scores))

    for i, idx in enumerate(hybrid_ranking[:5], 1):
        print(f"  {i}. doc_{idx:02d} (hybrid: {hybrid_scores[idx]:.4f})")
        print(f"     BM25: {bm25_scores[idx]:.1f}, Dense: {dense_scores[idx]:.4f}")
        print(f"     → {documents[idx]}")

    # Try different weights
    print(f"\n⚖️  Different Fusion Weights:")

    for weight in [0.3, 0.7]:
        hybrid_w = hybrid_score_mix(bm25_scores, dense_scores, weight=weight)
        top_idx = np.argmax(hybrid_w)
        print(f"\n   Weight={weight} (dense={weight}, sparse={1-weight}):")
        print(f"   Top result: doc_{top_idx:02d}")
        print(f"   → {documents[top_idx]}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Hybrid search combines complementary retrieval methods:")
    print("  - Sparse (BM25): exact term matching, good for keyword queries")
    print("  - Dense (embeddings): semantic matching, good for concept queries")
    print("  - Fusion: score = w·dense + (1-w)·sparse (after normalization)")
    print("\nBenefits:")
    print("  - Sparse catches exact term matches")
    print("  - Dense catches semantic similarity")
    print("  - Hybrid gets best of both!")
    print("\nUse cases: search engines, QA systems, document retrieval")

if __name__ == "__main__":
    main()
