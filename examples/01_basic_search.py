#!/usr/bin/env python3
"""
Example 1: Basic Vector Search
Demonstrates exact cosine similarity search using FlatIP index.
"""

import numpy as np
from embkit.lib.index.flatip import FlatIP
from embkit.lib.models.dummy import DummyEncoder

def main():
    print("=" * 60)
    print("Example 1: Basic Vector Search with FlatIP")
    print("=" * 60)

    # Create a simple encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create a small corpus
    documents = [
        "machine learning and artificial intelligence",
        "deep neural networks for computer vision",
        "natural language processing with transformers",
        "reinforcement learning for robotics",
        "data science and statistical analysis",
        "database management systems",
        "web development with javascript",
        "mobile app development",
        "cloud computing infrastructure",
        "cybersecurity and network protection"
    ]

    print(f"\n📚 Indexing {len(documents)} documents...")

    # Encode documents
    embeddings = encoder.encode_documents(documents)
    doc_ids = [f"doc_{i:03d}" for i in range(len(documents))]

    # Build index
    index = FlatIP(d=32)
    index.add(embeddings, doc_ids)
    print(f"✓ Index built with {len(doc_ids)} documents")

    # Search
    query = "artificial intelligence and neural networks"
    print(f"\n🔍 Query: '{query}'")

    q_vec = encoder.encode_queries([query])[0]
    result_ids, scores = index.search(q_vec, k=5)

    print(f"\n📊 Top 5 Results:")
    for i, (doc_id, score) in enumerate(zip(result_ids, scores), 1):
        idx = int(doc_id.split('_')[1])
        print(f"  {i}. {doc_id} (score: {score:.4f})")
        print(f"     → {documents[idx]}")

    print("\n✓ Example complete!")
    print("\nKey Concept: FlatIP provides exact cosine similarity search")
    print("via inner product on L2-normalized vectors.")

if __name__ == "__main__":
    main()
