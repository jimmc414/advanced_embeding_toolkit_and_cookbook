#!/usr/bin/env python3
"""
Example 3: Directional Search
Demonstrates shifting search results toward a desired direction vector.
Example: "neural networks" + direction("applications") → practical AI results
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.query_ops import directional
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 3: Directional Search")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create corpus with theory vs application documents
    documents = [
        "theoretical foundations of machine learning algorithms",
        "mathematical proofs in deep learning theory",
        "abstract neural network architectures",
        "applied machine learning in healthcare",
        "practical AI applications in finance",
        "real-world deployment of ML models",
        "theoretical computer science research",
        "production ML systems at scale",
        "academic papers on learning theory",
        "industry AI use cases and implementations"
    ]

    print(f"\n📚 Corpus: {len(documents)} documents")
    print("   (mix of theoretical and applied ML content)")

    # Encode
    embeddings = encoder.encode_documents(documents)
    doc_ids = [f"doc_{i:02d}" for i in range(len(documents))]

    # Create direction vector: "applied" - "theoretical"
    v_applied = encoder.encode_queries(["applied practical implementation"])[0]
    v_theory = encoder.encode_queries(["theoretical abstract research"])[0]
    v_direction = l2n(v_applied - v_theory, axis=None)

    # Query
    query = "machine learning systems"
    print(f"\n🔍 Query: '{query}'")
    q_vec = encoder.encode_queries([query])[0]

    # Standard search (alpha=0)
    print(f"\n📊 Standard Search (no direction):")
    ranked_standard = directional(q_vec, v_direction, embeddings, alpha=0.0)
    for i, (idx, score) in enumerate(ranked_standard[:5], 1):
        print(f"  {i}. doc_{idx:02d} (score: {score:.4f})")
        print(f"     → {documents[idx]}")

    # Directional search toward "applied"
    print(f"\n🎯 Directional Search (α=0.7 toward 'applied'):")
    ranked_directional = directional(q_vec, v_direction, embeddings, alpha=0.7)
    for i, (idx, score) in enumerate(ranked_directional[:5], 1):
        print(f"  {i}. doc_{idx:02d} (score: {score:.4f})")
        print(f"     → {documents[idx]}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Directional search computes similarity to:")
    print("  q' = normalize(q + α * v_direction)")
    print("This steers results toward the direction vector.")

if __name__ == "__main__":
    main()
