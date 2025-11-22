#!/usr/bin/env python3
"""
Example 19: Differential Privacy for Embeddings
Demonstrates adding calibrated noise for privacy protection.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.safety.privacy import dp_gaussian_noise
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 19: Differential Privacy Protection")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Sensitive user query
    sensitive_query = "private medical condition symptoms"
    print(f"\n🔒 Scenario: User submits sensitive query")
    print(f"   Query: '{sensitive_query}'")
    print(f"   Goal: Protect query privacy with differential privacy")

    # Original embedding
    original_vec = encoder.encode_queries([sensitive_query])[0]
    print(f"\n📊 Original embedding:")
    print(f"   Shape: {original_vec.shape}")
    print(f"   Norm: {np.linalg.norm(original_vec):.4f}")
    print(f"   Sample values: [{original_vec[0]:.4f}, {original_vec[1]:.4f}, ...]")

    # Apply DP noise with different privacy budgets
    print(f"\n🔐 Applying Differential Privacy...")

    epsilons = [0.1, 0.5, 1.0, 5.0]  # Privacy budgets (smaller = more private)
    delta = 1e-5  # Privacy parameter
    sensitivity = 2.0  # L2 sensitivity (max change from adding/removing one record)

    privatized_vecs = {}

    for epsilon in epsilons:
        # Add DP noise
        private_vec = dp_gaussian_noise(
            original_vec,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            clip_norm=1.0
        )
        privatized_vecs[epsilon] = private_vec

        # Measure utility loss
        similarity = float(l2n(original_vec, axis=None) @ l2n(private_vec, axis=None))
        noise_magnitude = np.linalg.norm(private_vec - original_vec)

        print(f"\n   ε={epsilon} (privacy budget):")
        print(f"      Similarity to original: {similarity:.4f}")
        print(f"      Noise magnitude: {noise_magnitude:.4f}")
        print(f"      Privacy guarantee: (ε={epsilon}, δ={delta})-DP")

        if epsilon < 1.0:
            print(f"      → Strong privacy, lower utility")
        elif epsilon < 3.0:
            print(f"      → Moderate privacy/utility tradeoff")
        else:
            print(f"      → Weaker privacy, better utility")

    # Search with privatized queries
    print(f"\n🔍 Search Results with Different Privacy Levels:")

    # Create corpus
    documents = [
        "medical symptoms and diagnosis",
        "health condition treatment options",
        "generic product information",
        "unrelated topic content",
        "clinical research findings",
    ]

    embeddings = encoder.encode_documents(documents)
    doc_ids = [f"doc_{i}" for i in range(len(documents))]

    # Original search
    print(f"\n   Original Query (no privacy):")
    Dn = l2n(embeddings, axis=1)
    orig_n = l2n(original_vec, axis=None)
    orig_scores = (Dn @ orig_n).astype(np.float32)
    orig_ranking = np.argsort(-orig_scores)

    for i, idx in enumerate(orig_ranking[:3], 1):
        print(f"      {i}. {doc_ids[idx]}: {documents[idx]}")
        print(f"         Score: {orig_scores[idx]:.4f}")

    # Privatized search (ε=1.0)
    print(f"\n   Privatized Query (ε=1.0):")
    private_vec = privatized_vecs[1.0]
    priv_n = l2n(private_vec, axis=None)
    priv_scores = (Dn @ priv_n).astype(np.float32)
    priv_ranking = np.argsort(-priv_scores)

    for i, idx in enumerate(priv_ranking[:3], 1):
        print(f"      {i}. {doc_ids[idx]}: {documents[idx]}")
        print(f"         Score: {priv_scores[idx]:.4f}")

    # Ranking stability
    print(f"\n📈 Privacy vs Utility Tradeoff:")
    print(f"   {'ε':>6s} {'Privacy':>12s} {'Similarity':>12s} {'Top-1 Match':>12s}")
    print(f"   {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

    for epsilon in epsilons:
        private_vec = privatized_vecs[epsilon]
        sim = float(l2n(original_vec, axis=None) @ l2n(private_vec, axis=None))

        # Check if top-1 is preserved
        priv_n = l2n(private_vec, axis=None)
        priv_scores = (Dn @ priv_n).astype(np.float32)
        top1_preserved = np.argmax(priv_scores) == orig_ranking[0]

        privacy_level = "Strong" if epsilon < 1 else ("Moderate" if epsilon < 3 else "Weak")

        print(f"   {epsilon:6.1f} {privacy_level:>12s} {sim:12.4f} {'✓' if top1_preserved else '✗':>12s}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Differential Privacy protects sensitive queries:")
    print("  1. Clip embedding to bounded sensitivity")
    print("  2. Add calibrated Gaussian noise: N(0, σ²)")
    print("  3. σ chosen based on privacy budget ε and δ")
    print("  4. Provides (ε,δ)-DP guarantee")
    print("\nTradeoff:")
    print("  - Smaller ε → stronger privacy, more noise, lower utility")
    print("  - Larger ε → weaker privacy, less noise, higher utility")
    print("\nUse cases: healthcare search, financial queries, personal data")

if __name__ == "__main__":
    main()
