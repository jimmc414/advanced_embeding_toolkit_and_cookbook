#!/usr/bin/env python3
"""
Example 4: Contrastive Search
Demonstrates finding results similar to query but dissimilar to a negative concept.
Example: "programming" BUT NOT "web development" → systems/scientific programming
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.query_ops import contrastive
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 4: Contrastive Search")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create corpus with different types of programming
    documents = [
        "web development with HTML CSS and JavaScript",
        "frontend frameworks like React and Vue",
        "systems programming in C and assembly",
        "scientific computing with Python and NumPy",
        "backend web services with Node.js",
        "operating systems and kernel development",
        "full-stack web applications",
        "numerical algorithms and linear algebra",
        "responsive web design and UI development",
        "high-performance computing and parallel algorithms"
    ]

    print(f"\n📚 Corpus: {len(documents)} documents")
    print("   (mix of web dev and systems/scientific programming)")

    # Encode
    embeddings = encoder.encode_documents(documents)
    doc_ids = [f"doc_{i:02d}" for i in range(len(documents))]

    # Query and negative concept
    query = "programming and software development"
    negative_concept = "web development frontend backend HTML CSS"

    print(f"\n🔍 Query: '{query}'")
    print(f"⛔ Negative: '{negative_concept}'")

    q_vec = encoder.encode_queries([query])[0]
    v_neg = encoder.encode_queries([negative_concept])[0]

    # Standard search
    print(f"\n📊 Standard Search:")
    ranked_standard = contrastive(q_vec, v_neg, embeddings, lam=0.0)
    for i, (idx, score) in enumerate(ranked_standard[:5], 1):
        print(f"  {i}. doc_{idx:02d} (score: {score:.4f})")
        print(f"     → {documents[idx]}")

    # Contrastive search
    print(f"\n🎯 Contrastive Search (λ=0.8, penalize web dev):")
    ranked_contrastive = contrastive(q_vec, v_neg, embeddings, lam=0.8)
    for i, (idx, score) in enumerate(ranked_contrastive[:5], 1):
        print(f"  {i}. doc_{idx:02d} (score: {score:.4f})")
        print(f"     → {documents[idx]}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Contrastive search computes:")
    print("  score = cos(q, d) - λ * cos(v_neg, d)")
    print("This promotes results similar to q but dissimilar to v_neg.")

if __name__ == "__main__":
    main()
