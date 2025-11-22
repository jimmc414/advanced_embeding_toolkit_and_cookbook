#!/usr/bin/env python3
"""
Example 9: Fairness-Aware Reranking
Demonstrates ensuring minimum representation of protected groups in search results.
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
from embkit.lib.query_ops import fair_rerank
from embkit.lib.utils import l2n

def main():
    print("=" * 60)
    print("Example 9: Fairness-Aware Reranking")
    print("=" * 60)

    # Create encoder
    encoder = DummyEncoder(d=32, seed=42)

    # Create corpus with group annotations
    candidates = [
        {"id": "doc_00", "text": "senior software engineer position", "group": "experienced", "score": 0.95},
        {"id": "doc_01", "text": "lead developer role", "group": "experienced", "score": 0.92},
        {"id": "doc_02", "text": "junior developer opening", "group": "entry-level", "score": 0.75},
        {"id": "doc_03", "text": "principal engineer role", "group": "experienced", "score": 0.90},
        {"id": "doc_04", "text": "entry-level programming job", "group": "entry-level", "score": 0.70},
        {"id": "doc_05", "text": "staff engineer position", "group": "experienced", "score": 0.88},
        {"id": "doc_06", "text": "graduate software role", "group": "entry-level", "score": 0.68},
        {"id": "doc_07", "text": "senior architect position", "group": "experienced", "score": 0.85},
        {"id": "doc_08", "text": "intern developer role", "group": "entry-level", "score": 0.65},
        {"id": "doc_09", "text": "engineering manager role", "group": "experienced", "score": 0.82},
    ]

    doc_ids = [c["id"] for c in candidates]
    relevance = {c["id"]: c["score"] for c in candidates}
    groups = {c["id"]: c["group"] for c in candidates}

    print(f"\n📚 Candidate pool: {len(candidates)} job postings")
    print(f"   - Experienced: {sum(1 for c in candidates if c['group'] == 'experienced')}")
    print(f"   - Entry-level: {sum(1 for c in candidates if c['group'] == 'entry-level')}")

    # Standard relevance-only ranking
    print(f"\n📊 Standard Ranking (by relevance only):")
    relevance_order = sorted(candidates, key=lambda x: -x["score"])[:5]
    for i, doc in enumerate(relevance_order, 1):
        print(f"  {i}. {doc['id']} (score: {doc['score']:.2f}, group: {doc['group']})")

    exp_count = sum(1 for d in relevance_order if d['group'] == 'experienced')
    entry_count = sum(1 for d in relevance_order if d['group'] == 'entry-level')
    print(f"   → Experienced: {exp_count}/5, Entry-level: {entry_count}/5")

    # Fair reranking with 40% minimum for entry-level
    print(f"\n⚖️  Fair Reranking (40% minimum for entry-level):")
    fair_ranking = fair_rerank(
        candidates=doc_ids,
        relevance=relevance,
        groups=groups,
        protected_group="entry-level",
        target_ratio=0.4,
        top_k=5
    )

    for i, doc_id in enumerate(fair_ranking, 1):
        doc = next(c for c in candidates if c["id"] == doc_id)
        print(f"  {i}. {doc_id} (score: {relevance[doc_id]:.2f}, group: {doc['group']})")

    exp_count_fair = sum(1 for doc_id in fair_ranking if groups[doc_id] == 'experienced')
    entry_count_fair = sum(1 for doc_id in fair_ranking if groups[doc_id] == 'entry-level')
    print(f"   → Experienced: {exp_count_fair}/5, Entry-level: {entry_count_fair}/5")
    print(f"   ✓ Achieved {entry_count_fair/5*100:.0f}% representation (target: 40%)")

    print("\n✓ Example complete!")
    print("\nKey Concept: Fair reranking ensures minimum group representation by:")
    print("  1. Computing required protected group items at each rank position")
    print("  2. Selecting highest-scoring protected items when quota not met")
    print("  3. Otherwise selecting by relevance")
    print("  4. Balancing fairness and relevance")
    print("\nUse cases: job search, candidate screening, content diversity")

if __name__ == "__main__":
    main()
