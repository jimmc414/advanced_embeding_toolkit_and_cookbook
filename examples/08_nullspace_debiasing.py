#!/usr/bin/env python3
"""
Example 8: Nullspace Projection for Debiasing
Demonstrates removing unwanted directions (e.g., gender bias) from embeddings.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.analysis.nullspace import remove_direction, remove_directions
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 8: Nullspace Projection for Debiasing")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Occupation words (potentially biased by gender in embeddings)
    occupations = [
        "doctor", "nurse", "engineer", "teacher",
        "programmer", "secretary", "CEO", "assistant"
    ]

    # Gender-defining word pairs
    male_words = ["he", "man", "male", "boy", "his", "himself"]
    female_words = ["she", "woman", "female", "girl", "her", "herself"]

    print("\n🔬 Computing gender direction from defining pairs...")

    # Encode gender words
    male_vecs = encoder.encode_documents(male_words)
    female_vecs = encoder.encode_documents(female_words)

    # Gender direction: average difference
    gender_direction = l2n((male_vecs.mean(axis=0) - female_vecs.mean(axis=0)), axis=None)

    print("✓ Gender direction computed")

    # Encode occupations
    occupation_vecs = encoder.encode_documents(occupations)

    # Measure gender bias before debiasing
    print(f"\n📊 Gender Bias (before debiasing):")
    for occ, vec in zip(occupations, occupation_vecs):
        bias = float(vec @ gender_direction)
        direction = "male" if bias > 0 else "female"
        print(f"  {occ:12s}: {bias:+.4f} ({direction})")

    # Remove gender direction
    print(f"\n🔧 Applying nullspace projection to remove gender...")
    debiased_vecs = remove_direction(occupation_vecs, gender_direction)

    print(f"\n📊 Gender Bias (after debiasing):")
    for occ, vec in zip(occupations, debiased_vecs):
        bias = float(vec @ gender_direction)
        print(f"  {occ:12s}: {bias:+.4f} (neutralized)")

    # Example with multiple directions
    print("\n" + "=" * 60)
    print("Removing Multiple Directions")
    print("=" * 60)

    # Add age direction
    young_words = ["young", "youth", "junior", "new"]
    old_words = ["old", "senior", "experienced", "veteran"]

    young_vecs = encoder.encode_documents(young_words)
    old_vecs = encoder.encode_documents(old_words)
    age_direction = l2n((young_vecs.mean(axis=0) - old_vecs.mean(axis=0)), axis=None)

    # Remove both gender and age
    directions = np.vstack([gender_direction, age_direction])
    multi_debiased = remove_directions(occupation_vecs, directions)

    print(f"\n🔧 Removed gender AND age directions")
    print(f"📊 Remaining bias:")
    for occ, vec in zip(occupations, multi_debiased):
        gender_bias = float(vec @ gender_direction)
        age_bias = float(vec @ age_direction)
        print(f"  {occ:12s}: gender={gender_bias:+.4f}, age={age_bias:+.4f}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Nullspace projection removes bias by:")
    print("  1. Identifying bias direction from defining word pairs")
    print("  2. Projecting embeddings onto subspace orthogonal to bias")
    print("  3. Preserving semantic meaning while removing unwanted correlations")
    print("\nMath: v' = v - (v·d)d  where d is the normalized bias direction")

if __name__ == "__main__":
    main()
