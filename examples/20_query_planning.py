#!/usr/bin/env python3
"""
Example 20: Query Planning and Multi-Hop Retrieval
Demonstrates decomposing complex queries into retrieval steps.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.planning import Document, planned_search, extract_born_in
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 20: Query Planning and Decomposition")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create knowledge corpus
    documents = [
        "George Washington was born in Virginia and served as the first US president",
        "Abraham Lincoln was born in Kentucky and led during the Civil War",
        "Franklin Roosevelt was born in New York and served four terms",
        "John Kennedy was born in Massachusetts and promoted the space program",
        "Ronald Reagan was born in California and was a former actor",
        "Virginia is known for its historical significance",
        "New York is a major financial and cultural center",
        "California is the most populous US state",
        "Massachusetts is home to many universities",
        "The Civil War reshaped American society",
    ]

    print(f"\n📚 Knowledge Base: {len(documents)} documents")

    # Encode
    embeddings = encoder.encode_documents(documents)

    # Create simple search function
    def search(query: str, k: int = 10):
        q_vec = encoder.encode_queries([query])[0]
        Dn = l2n(embeddings, axis=1)
        qn = l2n(q_vec, axis=None)
        scores = (Dn @ qn).astype(np.float32)
        ranking = np.argsort(-scores)[:k]

        results = []
        for idx in ranking:
            results.append(Document(
                id=f"doc_{idx:02d}",
                text=documents[idx],
                score=float(scores[idx])
            ))
        return results

    # Example 1: Simple query (no planning needed)
    print(f"\n" + "="*60)
    print("Example 1: Simple Query")
    print("="*60)

    simple_query = "US presidents"
    print(f"\n🔍 Query: '{simple_query}'")
    print(f"   → Direct retrieval (no decomposition needed)")

    results = search(simple_query, k=3)
    print(f"\n📊 Results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.id} (score: {doc.score:.4f})")
        print(f"     → {doc.text}")

    # Example 2: "Born in" query (requires filtering)
    print(f"\n" + "="*60)
    print("Example 2: Faceted Query with Planning")
    print("="*60)

    faceted_query = "presidents born in New York"
    print(f"\n🔍 Query: '{faceted_query}'")

    # Parse query
    parsed = extract_born_in(faceted_query)
    if parsed:
        entity_type, location = parsed
        print(f"   ✓ Parsed as faceted query:")
        print(f"      Entity: '{entity_type}'")
        print(f"      Location filter: '{location}'")

    # Use planning
    print(f"\n🧠 Query Plan:")
    print(f"   1. Retrieve candidates for 'presidents'")
    print(f"   2. Filter for mentions of 'New York'")

    planned_results = planned_search(faceted_query, search, k=5)

    print(f"\n📊 Planned Results:")
    for i, doc in enumerate(planned_results, 1):
        print(f"  {i}. {doc.id} (score: {doc.score:.4f})")
        print(f"     → {doc.text}")

    # Compare with naive search
    print(f"\n📊 Naive Direct Search (for comparison):")
    naive_results = search(faceted_query, k=3)
    for i, doc in enumerate(naive_results, 1):
        print(f"  {i}. {doc.id} (score: {doc.score:.4f})")
        print(f"     → {doc.text}")

    # Example 3: Custom planning
    print(f"\n" + "="*60)
    print("Example 3: Custom Multi-Step Planning")
    print("="*60)

    complex_query = "actors who became president"
    print(f"\n🔍 Query: '{complex_query}'")

    print(f"\n🧠 Query Plan:")
    print(f"   1. Search for 'president' to get candidates")
    print(f"   2. Filter for mentions of 'actor'")

    # Step 1: Get president candidates
    step1_results = search("president", k=10)
    print(f"\n   Step 1: Retrieved {len(step1_results)} president candidates")

    # Step 2: Filter for actor mentions
    filtered = [doc for doc in step1_results if "actor" in doc.text.lower()]

    print(f"\n📊 Final Results (after filtering):")
    if filtered:
        for i, doc in enumerate(filtered, 1):
            print(f"  {i}. {doc.id} (score: {doc.score:.4f})")
            print(f"     → {doc.text}")
    else:
        print(f"  No exact matches found")
        print(f"\n  Top candidates from step 1:")
        for i, doc in enumerate(step1_results[:3], 1):
            print(f"  {i}. {doc.id}: {doc.text}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Query planning decomposes complex queries:")
    print("  1. Parse query to identify structure (facets, constraints)")
    print("  2. Plan retrieval strategy (retrieve, filter, rerank)")
    print("  3. Execute plan steps sequentially")
    print("  4. Combine results intelligently")
    print("\nBenefits:")
    print("  - Handles complex multi-part queries")
    print("  - More accurate than single-step retrieval")
    print("  - Supports compositional reasoning")
    print("\nUse cases: QA systems, knowledge graphs, complex search")

if __name__ == "__main__":
    main()
