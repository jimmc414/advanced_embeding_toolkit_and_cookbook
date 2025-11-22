#!/usr/bin/env python3
"""
Example 15: Counterfactual Analysis
Demonstrates analyzing how search results change when swapping query facets.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.analysis.counterfactual import generate_counterfactuals, rank_delta
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 15: Counterfactual Query Analysis")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create corpus
    documents = [
        "Python web development with Django",
        "Java enterprise applications",
        "Python data science libraries",
        "JavaScript frontend frameworks",
        "Python machine learning tools",
        "Java Android app development",
        "Python web scraping tutorials",
        "Ruby on Rails web development",
        "Python automation scripts",
        "Java Spring Boot framework",
    ]

    print(f"\n📚 Corpus: {len(documents)} documents")

    # Encode
    embeddings = encoder.encode_documents(documents)
    doc_ids = [f"doc_{i:02d}" for i in range(len(documents))]

    # Original query
    original_query = "Python web development"
    print(f"\n🔍 Original Query: '{original_query}'")

    # Search with original query
    q_vec = encoder.encode_queries([original_query])[0]
    Dn = l2n(embeddings, axis=1)
    qn = l2n(q_vec, axis=None)
    scores = (Dn @ qn).astype(np.float32)
    original_ranking = np.argsort(-scores)

    print(f"\n📊 Original Results:")
    for i, idx in enumerate(original_ranking[:5], 1):
        print(f"  {i}. {doc_ids[idx]}: {documents[idx]}")
        print(f"      Score: {scores[idx]:.4f}")

    # Define facet alternatives for counterfactual generation
    facet_map = {
        "Python": ["Java", "JavaScript", "Ruby"],
        "web": ["data", "machine learning", "automation"],
    }

    print(f"\n🔄 Generating counterfactual queries...")
    print(f"   Facet variations:")
    for facet, alternatives in facet_map.items():
        print(f"   - '{facet}' → {alternatives}")

    counterfactuals = generate_counterfactuals(original_query, facet_map)

    print(f"\n📝 Generated {len(counterfactuals)} counterfactual queries:")
    for cf in counterfactuals:
        print(f"   - {cf}")

    # Analyze each counterfactual
    print(f"\n🔬 Counterfactual Analysis:")

    for cf_query in counterfactuals[:3]:  # Show first 3
        print(f"\n   Counterfactual: '{cf_query}'")

        # Search with counterfactual
        cf_vec = encoder.encode_queries([cf_query])[0]
        cf_n = l2n(cf_vec, axis=None)
        cf_scores = (Dn @ cf_n).astype(np.float32)
        cf_ranking = np.argsort(-cf_scores)

        print(f"   Top results:")
        for i, idx in enumerate(cf_ranking[:3], 1):
            print(f"      {i}. {doc_ids[idx]}: {documents[idx]}")

        # Compute rank changes
        original_ids = [doc_ids[i] for i in original_ranking]
        cf_ids = [doc_ids[i] for i in cf_ranking]
        deltas = rank_delta(original_ids, cf_ids)

        # Show biggest changes
        significant_changes = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        print(f"\n   Biggest rank changes:")
        for doc_id, delta in significant_changes:
            if delta != 0:
                direction = "↑" if delta > 0 else "↓"
                idx = int(doc_id.split('_')[1])
                print(f"      {doc_id}: {direction}{abs(delta)} positions - {documents[idx]}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Counterfactual analysis reveals query sensitivity:")
    print("  1. Generate variations by swapping facets/attributes")
    print("  2. Compare rankings across original and counterfactuals")
    print("  3. Identify which documents are sensitive to specific facets")
    print("  4. Understand model behavior and potential biases")
    print("\nUse cases:")
    print("  - Fairness testing (e.g., gender swaps)")
    print("  - Query understanding (which facets matter most)")
    print("  - Model debugging and bias detection")

if __name__ == "__main__":
    main()
