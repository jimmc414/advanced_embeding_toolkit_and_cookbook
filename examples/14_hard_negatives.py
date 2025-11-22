#!/usr/bin/env python3
"""
Example 14: Hard Negative Mining
Demonstrates finding challenging negative examples for training.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.training.hard_negative import mine_hard_negatives, triplet_margin
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 14: Hard Negative Mining for Training")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Training scenario: query and candidate documents
    query_text = "machine learning algorithms"

    # Positive examples (relevant to query)
    positive_docs = [
        "supervised learning classification methods",
        "neural network architectures",
    ]

    # Candidate negative examples (varying difficulty)
    candidate_docs = [
        # Easy negatives (clearly different)
        "cooking recipes and food preparation",
        "gardening tips for beginners",
        "travel destinations in Europe",

        # Medium difficulty
        "computer science fundamentals",
        "programming language design",

        # Hard negatives (topically close but not relevant)
        "statistical data analysis techniques",
        "deep learning optimization strategies",
        "artificial intelligence research papers",
        "predictive modeling frameworks",
        "data mining and analytics",
    ]

    print(f"\n📚 Training data:")
    print(f"   Query: '{query_text}'")
    print(f"   Positives: {len(positive_docs)} examples")
    print(f"   Candidate negatives: {len(candidate_docs)} examples")

    # Encode
    q_vec = encoder.encode_queries([query_text])[0]
    pos_vecs = encoder.encode_documents(positive_docs)
    cand_vecs = encoder.encode_documents(candidate_docs)

    # Rank candidates by similarity to query
    qn = l2n(q_vec, axis=None)
    cand_normed = l2n(cand_vecs, axis=1)
    scores = (cand_normed @ qn).astype(np.float32)

    # Sort by score (descending)
    ranked_indices = np.argsort(-scores)
    ranked_docs = [candidate_docs[i] for i in ranked_indices]

    print(f"\n📊 Candidates ranked by similarity to query:")
    for i, (idx, score) in enumerate(zip(ranked_indices, scores[ranked_indices]), 1):
        difficulty = "🔴 HARD" if score > 0.3 else ("🟡 MEDIUM" if score > 0.15 else "🟢 EASY")
        print(f"  {i:2d}. {difficulty} (score: {score:.4f})")
        print(f"      → {candidate_docs[idx]}")

    # Mine hard negatives (top-k most similar)
    print(f"\n⛏️  Mining top-3 hard negatives...")
    positive_set = set()  # Empty for this example (all candidates are negatives)
    hard_negatives = mine_hard_negatives(ranked_docs, positive_set, limit=3)

    print(f"\n🎯 Selected hard negatives for training:")
    for i, neg in enumerate(hard_negatives, 1):
        idx = candidate_docs.index(neg)
        print(f"  {i}. {neg}")
        print(f"     Similarity to query: {scores[idx]:.4f}")

    # Compute triplet loss for each hard negative
    print(f"\n📐 Triplet Margin Loss (margin=0.2):")
    print(f"   (goal: positive should be closer than negative by margin)")

    pos_vec = pos_vecs[0]  # Use first positive
    for i, neg in enumerate(hard_negatives, 1):
        idx = candidate_docs.index(neg)
        neg_vec = cand_vecs[idx]
        loss = triplet_margin(q_vec, pos_vec, neg_vec, margin=0.2)

        pos_sim = float(l2n(q_vec, axis=None) @ l2n(pos_vec, axis=None))
        neg_sim = float(l2n(q_vec, axis=None) @ l2n(neg_vec, axis=None))

        print(f"\n   Hard negative {i}:")
        print(f"      Positive similarity: {pos_sim:.4f}")
        print(f"      Negative similarity: {neg_sim:.4f}")
        print(f"      Triplet loss: {loss:.4f}")

        if loss > 0:
            print(f"      ⚠️  Violation! Negative too close (needs training)")
        else:
            print(f"      ✓ Margin satisfied")

    print("\n✓ Example complete!")
    print("\nKey Concept: Hard negative mining improves model training by:")
    print("  1. Ranking negatives by similarity to query")
    print("  2. Selecting hardest (most similar) negatives")
    print("  3. Training with triplet loss: L = max(0, margin + sim(q,neg) - sim(q,pos))")
    print("  4. Forces model to learn fine-grained distinctions")
    print("\nWhy it works: Easy negatives don't provide learning signal")
    print("Use cases: metric learning, embedding fine-tuning, ranking models")

if __name__ == "__main__":
    main()
