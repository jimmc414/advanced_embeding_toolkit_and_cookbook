#!/usr/bin/env python3
"""
Example 11: Personalized Search
Demonstrates user-specific ranking using user profile vectors.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.query_ops import personalize
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 11: Personalized Search")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create diverse documents
    documents = [
        "beginner guide to Python programming",
        "advanced machine learning algorithms",
        "introduction to data visualization",
        "deep dive into neural networks",
        "basic web development tutorial",
        "expert-level distributed systems",
        "getting started with databases",
        "advanced computer vision techniques",
    ]

    print(f"\n📚 Corpus: {len(documents)} documents")
    print("   (mix of beginner and advanced content)")

    # Encode
    embeddings = encoder.encode_documents(documents)

    # Create two user profiles
    # User 1: Beginner interested in learning basics
    beginner_history = [
        "introduction to programming",
        "basic tutorials",
        "getting started guides"
    ]

    # User 2: Expert interested in advanced topics
    expert_history = [
        "advanced algorithms",
        "expert techniques",
        "deep technical content"
    ]

    beginner_vecs = encoder.encode_documents(beginner_history)
    expert_vecs = encoder.encode_documents(expert_history)

    user_beginner = l2n(beginner_vecs.mean(axis=0), axis=None)
    user_expert = l2n(expert_vecs.mean(axis=0), axis=None)

    # Query (same for both users)
    query = "programming tutorial"
    print(f"\n🔍 Query: '{query}'")
    q_vec = encoder.encode_queries([query])[0]

    # Generic (non-personalized) ranking
    print(f"\n📊 Generic Ranking (no personalization):")
    Dn = l2n(embeddings, axis=1)
    qn = l2n(q_vec, axis=None)
    generic_scores = (Dn @ qn).astype(np.float32)
    generic_order = np.argsort(-generic_scores)
    for i, idx in enumerate(generic_order[:5], 1):
        print(f"  {i}. {documents[idx]}")
        print(f"      Score: {generic_scores[idx]:.4f}")

    # Personalized for beginner user
    print(f"\n👤 Personalized for Beginner User (β=0.3):")
    beginner_scores = personalize(q_vec, user_beginner, embeddings, beta=0.3)
    beginner_order = np.argsort(-beginner_scores)
    for i, idx in enumerate(beginner_order[:5], 1):
        print(f"  {i}. {documents[idx]}")
        print(f"      Score: {beginner_scores[idx]:.4f} (base: {generic_scores[idx]:.4f})")

    # Personalized for expert user
    print(f"\n👤 Personalized for Expert User (β=0.3):")
    expert_scores = personalize(q_vec, user_expert, embeddings, beta=0.3)
    expert_order = np.argsort(-expert_scores)
    for i, idx in enumerate(expert_order[:5], 1):
        print(f"  {i}. {documents[idx]}")
        print(f"      Score: {expert_scores[idx]:.4f} (base: {generic_scores[idx]:.4f})")

    print("\n✓ Example complete!")
    print("\nKey Concept: Personalization combines query and user profile:")
    print("  score' = cos(q, d) + β × cos(user_profile, d)")
    print("  - user_profile = aggregate of user's historical interactions")
    print("  - β controls personalization strength")
    print("  - Same query, different results per user!")
    print("\nUse cases: e-commerce, content recommendations, job search")

if __name__ == "__main__":
    main()
