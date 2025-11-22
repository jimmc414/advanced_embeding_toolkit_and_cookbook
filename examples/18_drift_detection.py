#!/usr/bin/env python3
"""
Example 18: Distribution Drift Detection
Demonstrates detecting when query distribution shifts using Hotelling's T²
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.monitoring.drift import hotelling_t2, detect_drift

def main():
    print("=" * 60)
    print("Example 18: Query Distribution Drift Detection")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=16, seed=42)

    # Simulate reference distribution (historical queries)
    print("\n📊 Building reference distribution...")
    print("   (Based on 1 month of historical queries)")

    reference_queries = [
        "machine learning tutorial",
        "python programming guide",
        "data science basics",
        "neural networks introduction",
        "statistics for data analysis",
        "deep learning course",
        "pandas dataframe operations",
        "supervised learning algorithms",
        "numpy array manipulation",
        "feature engineering methods",
    ] * 5  # Repeat for more samples

    ref_embeddings = encoder.encode_documents(reference_queries)

    # Compute reference statistics
    ref_mean = ref_embeddings.mean(axis=0)
    ref_cov = np.cov(ref_embeddings, rowvar=False)

    print(f"   Reference samples: {len(reference_queries)}")
    print(f"   Mean vector: shape {ref_mean.shape}")
    print(f"   Covariance: shape {ref_cov.shape}")

    # Test batch 1: Similar distribution (no drift)
    print(f"\n✅ Test Batch 1: Similar queries (expecting NO drift)")
    similar_queries = [
        "learn machine learning",
        "python coding tutorial",
        "introduction to data science",
        "neural network basics",
        "statistical analysis guide",
    ]

    similar_embeddings = encoder.encode_documents(similar_queries)
    t2_similar = hotelling_t2(similar_embeddings, ref_mean, ref_cov, regularizer=1e-6)

    print(f"   Queries: {len(similar_queries)} samples")
    print(f"   Hotelling's T²: {t2_similar:.2f}")

    threshold = 30.0  # Example threshold
    is_drift_similar = detect_drift(similar_embeddings, ref_mean, ref_cov, threshold=threshold)
    print(f"   Drift detected (T² > {threshold}): {is_drift_similar}")

    # Test batch 2: Different distribution (drift!)
    print(f"\n⚠️  Test Batch 2: Different topic queries (expecting DRIFT)")
    different_queries = [
        "cooking recipes for dinner",
        "travel destinations in Europe",
        "car maintenance tips",
        "gardening advice for spring",
        "home workout exercises",
    ]

    different_embeddings = encoder.encode_documents(different_queries)
    t2_different = hotelling_t2(different_embeddings, ref_mean, ref_cov, regularizer=1e-6)

    print(f"   Queries: {len(different_queries)} samples")
    print(f"   Hotelling's T²: {t2_different:.2f}")

    is_drift_different = detect_drift(different_embeddings, ref_mean, ref_cov, threshold=threshold)
    print(f"   Drift detected (T² > {threshold}): {is_drift_different}")

    # Comparison
    print(f"\n📈 Statistical Analysis:")
    print(f"   {'Batch':20s} {'T² Statistic':>15s} {'Status':>15s}")
    print(f"   {'-'*20} {'-'*15} {'-'*15}")
    print(f"   {'Similar queries':20s} {t2_similar:15.2f} {'✓ No drift':>15s}")
    print(f"   {'Different topics':20s} {t2_different:15.2f} {'⚠ DRIFT!':>15s}")
    print(f"   {'Threshold':20s} {threshold:15.2f} {'-':>15s}")

    # Gradual drift simulation
    print(f"\n📊 Simulating Gradual Drift Over Time:")
    drift_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for ratio in drift_ratios:
        # Mix similar and different queries
        n_similar = int(10 * (1 - ratio))
        n_different = int(10 * ratio)

        mixed_queries = (similar_queries[:n_similar] +
                        different_queries[:n_different])

        if len(mixed_queries) > 0:
            mixed_embeddings = encoder.encode_documents(mixed_queries)
            t2_mixed = hotelling_t2(mixed_embeddings, ref_mean, ref_cov, regularizer=1e-6)
            status = "DRIFT" if t2_mixed > threshold else "OK"

            print(f"   {ratio*100:3.0f}% drift: T²={t2_mixed:6.2f} [{status}]")

    print("\n✓ Example complete!")
    print("\nKey Concept: Hotelling's T² detects distribution shift:")
    print("  T² = (μ_batch - μ_ref)ᵀ Σ⁻¹ (μ_batch - μ_ref)")
    print("  - Measures how far batch mean is from reference")
    print("  - Accounts for covariance structure")
    print("  - Alert when T² exceeds threshold")
    print("\nUse cases:")
    print("  - Monitor query distribution changes")
    print("  - Detect seasonal trends")
    print("  - Alert for model retraining")
    print("  - Quality assurance for production systems")

if __name__ == "__main__":
    main()
