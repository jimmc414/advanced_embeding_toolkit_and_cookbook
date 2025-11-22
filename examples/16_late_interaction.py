#!/usr/bin/env python3
"""
Example 16: Late Interaction (ColBERT-style)
Demonstrates multi-vector token-level interaction for fine-grained matching.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.query_ops import late_interaction_score
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 16: Late Interaction (ColBERT-style)")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=16, seed=42)  # Smaller dim for clarity

    # Simulate multi-vector representations
    # In real ColBERT: each token gets its own vector
    # Here: we'll create synthetic multi-vector representations

    print("\n📝 Scenario: Query with multiple concepts")
    print("   Query: 'machine learning neural networks'")
    print("   (3 tokens → 3 query vectors)")

    # Query tokens (3 vectors, one per term)
    query_terms = ["machine", "learning", "neural"]
    query_vecs = encoder.encode_documents(query_terms)
    print(f"   Query representation: {query_vecs.shape}")

    # Documents with varying term coverage
    docs = [
        {
            "text": "machine learning algorithms and methods",
            "tokens": ["machine", "learning", "algorithms", "methods"]
        },
        {
            "text": "neural network architectures",
            "tokens": ["neural", "network", "architectures"]
        },
        {
            "text": "deep neural networks for machine learning",
            "tokens": ["deep", "neural", "networks", "machine", "learning"]
        },
        {
            "text": "data science and statistics",
            "tokens": ["data", "science", "statistics"]
        }
    ]

    print(f"\n📚 Documents: {len(docs)} items")

    # Encode each document's tokens as multiple vectors
    doc_representations = []
    for doc in docs:
        doc_vecs = encoder.encode_documents(doc["tokens"])
        doc_representations.append(doc_vecs)

    # Compute late interaction scores
    print(f"\n🔍 Computing Late Interaction Scores...")
    print(f"   (MaxSim: for each query token, find max similarity to any doc token)")

    scores = []
    for i, (doc, doc_vecs) in enumerate(zip(docs, doc_representations)):
        score = late_interaction_score(query_vecs, doc_vecs)
        scores.append((i, score))

        print(f"\n   Doc {i}: '{doc['text']}'")
        print(f"   Tokens: {doc['tokens']}")
        print(f"   Shape: {doc_vecs.shape}")

        # Show per-query-term max matches
        qn = l2n(query_vecs, axis=1)
        dn = l2n(doc_vecs, axis=1)

        print(f"   Per-term matches:")
        for q_idx, q_term in enumerate(query_terms):
            sims = dn @ qn[q_idx]
            max_sim = np.max(sims)
            best_doc_token_idx = np.argmax(sims)
            best_doc_token = doc["tokens"][best_doc_token_idx]
            print(f"      '{q_term}' → best match: '{best_doc_token}' (sim: {max_sim:.4f})")

        print(f"   → Total Score: {score:.4f}")

    # Rank documents
    scores.sort(key=lambda x: -x[1])

    print(f"\n📊 Final Ranking:")
    for rank, (doc_idx, score) in enumerate(scores, 1):
        print(f"  {rank}. Doc {doc_idx} (score: {score:.4f})")
        print(f"     → {docs[doc_idx]['text']}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Late interaction enables fine-grained token matching:")
    print("  1. Represent query as multiple vectors (one per token)")
    print("  2. Represent document as multiple vectors (one per token)")
    print("  3. For each query token, find max similarity to any doc token")
    print("  4. Sum these max similarities")
    print("\nFormula: Score = Σ_q max_d sim(q_token, d_token)")
    print("\nAdvantages:")
    print("  - Captures term-level matches")
    print("  - Better than single-vector for multi-concept queries")
    print("  - Used in ColBERT and similar models")

if __name__ == "__main__":
    main()
