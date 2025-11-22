#!/usr/bin/env python3
"""
Example 2: Approximate Nearest Neighbor Search
Demonstrates fast approximate search using IVF-PQ (Inverted File with Product Quantization).
"""

import numpy as np
from embkit.lib.index.ivfpq import IVFPQ
from embkit.lib.models.dummy import DummyEncoder

def main():
    print("=" * 60)
    print("Example 2: Approximate Search with IVF-PQ")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=64, seed=123)

    # Create a larger corpus for ANN
    print("\n📚 Generating 1000 synthetic documents...")
    documents = []
    for i in range(1000):
        topic = i % 10
        topics = ["AI", "Database", "Web", "Mobile", "Cloud",
                  "Security", "Data Science", "DevOps", "IoT", "Blockchain"]
        documents.append(f"Document {i} about {topics[topic]} technology")

    # Encode
    embeddings = encoder.encode_documents(documents)
    doc_ids = [f"doc_{i:04d}" for i in range(len(documents))]

    # Build IVF-PQ index
    print("🔨 Building IVF-PQ index (nlist=32, m=8, nbits=8)...")
    index = IVFPQ(d=64, nlist=32, m=8, nbits=8, nprobe=8)
    index.train_add(embeddings, doc_ids)
    print("✓ Index built and trained")

    # Search
    query = "artificial intelligence and blockchain"
    print(f"\n🔍 Query: '{query}'")

    q_vec = encoder.encode_queries([query])[0]
    result_ids, scores = index.search(q_vec, k=10)

    print(f"\n📊 Top 10 Results:")
    for i, (doc_id, score) in enumerate(zip(result_ids, scores), 1):
        idx = int(doc_id.split('_')[1])
        print(f"  {i}. {doc_id} (score: {score:.4f}) → {documents[idx]}")

    print("\n✓ Example complete!")
    print("\nKey Concept: IVF-PQ trades accuracy for speed using:")
    print("  - Inverted File: coarse quantization into clusters")
    print("  - Product Quantization: compression via subspace quantization")
    print("  - This example re-ranks top candidates with exact scores")

if __name__ == "__main__":
    main()
