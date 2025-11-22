# 📚 Examples Guide

This directory contains **20 standalone, working examples** demonstrating every major concept in the Advanced Vector-Embedding Operations Toolkit. Each example is a drop-in Python script that runs independently without any setup.

## 🚀 Quick Start

### Method 1: Using the Helper Script (Recommended)

```bash
# Run a single example by number
./examples/run_example.sh 01

# Run a single example by name
./examples/run_example.sh 01_basic_search.py

# Run all examples
./examples/run_example.sh all
```

### Method 2: Direct Python Execution

```bash
# Set PYTHONPATH and run
export PYTHONPATH=$(pwd):$PYTHONPATH
python examples/01_basic_search.py
python examples/10_temporal_decay.py
python examples/15_counterfactual.py
```

## 📖 Examples by Category

### 🔍 **Core Search Operations**

#### 01. Basic Vector Search
**File:** `01_basic_search.py`
**Concept:** Exact cosine similarity search using FlatIP index
**Key Learning:** Foundation of vector search - inner product on L2-normalized vectors

#### 02. Approximate Search (IVF-PQ)
**File:** `02_approximate_search.py`
**Concept:** Fast approximate nearest neighbors for large-scale search
**Key Learning:** Trade accuracy for speed using inverted file + product quantization

---

### 🎯 **Query Operations**

#### 03. Directional Search
**File:** `03_directional_search.py`
**Concept:** Steer results toward a desired direction (e.g., "applied" vs "theoretical")
**Key Learning:** `q' = normalize(q + α * v_direction)` shifts ranking
**Use Case:** Bias search toward specific attributes

#### 04. Contrastive Search
**File:** `04_contrastive_search.py`
**Concept:** Find results similar to query BUT dissimilar to negative concept
**Key Learning:** `score = cos(q,d) - λ * cos(v_neg,d)` penalizes unwanted content
**Use Case:** "programming BUT NOT web development"

#### 05. MMR (Diversity)
**File:** `05_mmr_diversity.py`
**Concept:** Maximal Marginal Relevance for diverse, non-redundant results
**Key Learning:** Balance relevance and diversity iteratively
**Use Case:** Avoid duplicate or near-duplicate results

#### 10. Temporal Decay
**File:** `10_temporal_decay.py`
**Concept:** Time-based scoring that prefers recent documents
**Key Learning:** `score' = score × exp(-γ × age)` exponentially decays old content
**Use Case:** News search, trending topics

#### 11. Personalization
**File:** `11_personalization.py`
**Concept:** User-specific ranking based on profile vectors
**Key Learning:** `score' = cos(q,d) + β × cos(user_profile,d)`
**Use Case:** E-commerce, content recommendations

---

### 🛡️ **Safety & Privacy**

#### 06. PII Filtering
**File:** `06_pii_filtering.py`
**Concept:** Detect and redact personally identifiable information
**Key Learning:** Regex-based detection of emails, phones, SSNs
**Use Case:** Prevent accidental PII exposure in search results

#### 07. Repellor Vectors
**File:** `07_repellors.py`
**Concept:** Penalize toxic/unwanted content using repellor vectors
**Key Learning:** `score' = score - λ * max_b cos(d, repellor_b)`
**Use Case:** Content moderation, brand safety

#### 19. Differential Privacy
**File:** `19_privacy_dp.py`
**Concept:** Protect query privacy with calibrated noise
**Key Learning:** Add Gaussian noise for (ε,δ)-DP guarantee
**Use Case:** Healthcare search, sensitive queries

---

### ⚖️ **Fairness & Debiasing**

#### 08. Nullspace Debiasing
**File:** `08_nullspace_debiasing.py`
**Concept:** Remove unwanted bias directions (e.g., gender) from embeddings
**Key Learning:** Project onto orthogonal subspace: `v' = v - (v·d)d`
**Use Case:** Fair ranking, debiasing word embeddings

#### 09. Fairness Reranking
**File:** `09_fairness_reranking.py`
**Concept:** Ensure minimum representation of protected groups
**Key Learning:** Constrained optimization balancing fairness and relevance
**Use Case:** Job search, candidate screening

---

### 🔬 **Advanced Retrieval**

#### 12. Graph-Based Search
**File:** `12_graph_search.py`
**Concept:** kNN graph + Personalized PageRank for query expansion
**Key Learning:** PPR propagates relevance through document graph
**Use Case:** Related documents, recommendation

#### 16. Late Interaction (ColBERT)
**File:** `16_late_interaction.py`
**Concept:** Multi-vector token-level matching
**Key Learning:** `Score = Σ_q max_d sim(q_token, d_token)`
**Use Case:** Fine-grained term matching, multi-concept queries

#### 17. Hybrid Search
**File:** `17_hybrid_fusion.py`
**Concept:** Combine sparse (BM25) and dense (embedding) retrieval
**Key Learning:** Fuse complementary signals for better coverage
**Use Case:** Search engines, document retrieval

#### 20. Query Planning
**File:** `20_query_planning.py`
**Concept:** Decompose complex queries into multi-step retrieval
**Key Learning:** Parse, plan, execute sequential retrieval steps
**Use Case:** QA systems, compositional queries

---

### 📊 **Model Quality & Training**

#### 13. Calibration
**File:** `13_calibration.py`
**Concept:** Temperature scaling for well-calibrated confidence scores
**Key Learning:** `p = sigmoid(logits / T)` improves probability estimates
**Use Case:** Confidence thresholds, uncertainty estimation

#### 14. Hard Negative Mining
**File:** `14_hard_negatives.py`
**Concept:** Select challenging negatives for training
**Key Learning:** Triplet loss on hardest negatives improves discrimination
**Use Case:** Metric learning, embedding fine-tuning

#### 18. Drift Detection
**File:** `18_drift_detection.py`
**Concept:** Detect query distribution shifts using Hotelling's T²
**Key Learning:** Monitor for production degradation
**Use Case:** Model monitoring, retraining triggers

---

### 🧪 **Analysis & Debugging**

#### 15. Counterfactual Analysis
**File:** `15_counterfactual.py`
**Concept:** Analyze how results change when swapping query facets
**Key Learning:** Identify model sensitivities and biases
**Use Case:** Fairness testing, model debugging

---

## 🎓 Learning Path

### **Beginner Track**
Start with core concepts:
1. Example 01: Basic Search
2. Example 02: Approximate Search
3. Example 05: MMR Diversity
4. Example 06: PII Filtering

### **Intermediate Track**
Advanced query operations:
1. Example 03: Directional Search
2. Example 04: Contrastive Search
3. Example 10: Temporal Decay
4. Example 11: Personalization
5. Example 17: Hybrid Search

### **Advanced Track**
Research-level techniques:
1. Example 08: Nullspace Debiasing
2. Example 12: Graph-Based Search
3. Example 13: Calibration
4. Example 14: Hard Negative Mining
5. Example 16: Late Interaction
6. Example 18: Drift Detection
7. Example 19: Differential Privacy
8. Example 20: Query Planning

### **Safety & Fairness Track**
Responsible AI:
1. Example 06: PII Filtering
2. Example 07: Repellor Vectors
3. Example 08: Nullspace Debiasing
4. Example 09: Fairness Reranking
5. Example 15: Counterfactual Analysis
6. Example 19: Differential Privacy

---

## 💡 Key Concepts Reference

| Concept | Formula | Use Case |
|---------|---------|----------|
| **Cosine Similarity** | `cos(q,d) = q·d / (‖q‖‖d‖)` | Basic relevance |
| **Directional** | `score(q+αv, d)` | Attribute steering |
| **Contrastive** | `cos(q,d) - λ·cos(v_neg,d)` | Negative filtering |
| **MMR** | `λ·rel - (1-λ)·div` | Diversity |
| **Temporal** | `score·exp(-γt)` | Recency bias |
| **Personalization** | `cos(q,d) + β·cos(u,d)` | User preferences |
| **Repellor** | `score - λ·max_b cos(d,b)` | Safety |
| **Nullspace** | `v - (v·d)d` | Debiasing |
| **PPR** | `α·seed + (1-α)P^T r` | Graph expansion |
| **Temperature** | `sigmoid(z/T)` | Calibration |
| **DP Noise** | `v + N(0,σ²I)` | Privacy |

---

## 🏃 Running Examples

### Individual Example
```bash
python examples/03_directional_search.py
```

### All Examples
```bash
# Bash
for f in examples/*.py; do
    echo "Running $f..."
    python "$f" || echo "Failed: $f"
done

# Or use Python
python -c "
import os, subprocess
for f in sorted(os.listdir('examples')):
    if f.endswith('.py'):
        print(f'\\n{"="*60}\\nRunning {f}\\n{"="*60}')
        subprocess.run(['python', f'examples/{f}'])
"
```

---

## 🔧 Requirements

All examples use the **DummyEncoder** by default, which requires only:
- numpy
- scipy
- scikit-learn
- faiss-cpu
- networkx
- pydantic
- pyyaml

Install with:
```bash
pip install numpy scipy scikit-learn faiss-cpu networkx pydantic pyyaml
```

**Optional:** For Hugging Face models (not used in examples):
```bash
pip install torch transformers sentence-transformers huggingface-hub
```

---

## 📝 Example Template

Each example follows this structure:

```python
#!/usr/bin/env python3
"""
Example N: Title
Brief description of the concept
"""

import numpy as np
from embkit.lib.models.dummy import DummyEncoder
# ... other imports

def main():
    print("=" * 60)
    print("Example N: Title")
    print("=" * 60)

    # 1. Setup
    encoder = DummyEncoder(d=32, seed=42)

    # 2. Create data
    documents = [...]

    # 3. Demonstrate concept
    # ... core logic ...

    # 4. Show results
    print(f"\n📊 Results:")
    # ...

    # 5. Explain key concept
    print("\n✓ Example complete!")
    print("\nKey Concept: ...")
    print("Use cases: ...")

if __name__ == "__main__":
    main()
```

---

## 🤝 Contributing

To add a new example:

1. Create `examples/NN_concept_name.py`
2. Follow the template structure
3. Include:
   - Clear title and description
   - Self-contained data generation
   - Informative output with emojis
   - "Key Concept" explanation
   - Practical use cases
4. Test: `python examples/NN_concept_name.py`
5. Update this README

---

## 📚 Further Reading

- **Architecture:** See `architecture.md` for system design
- **Implementation:** See `implementation.md` for code details
- **API Reference:** Check docstrings in `embkit/lib/`
- **Research Papers:** See `README.md` for citations

---

## ✨ Features

✅ **Zero setup required** - all examples generate their own data
✅ **Standalone scripts** - run independently, no dependencies between examples
✅ **Clear output** - emojis and formatting for easy reading
✅ **Educational** - each explains the "why" not just the "how"
✅ **Production-ready patterns** - real-world applicable code
✅ **Comprehensive** - covers 20+ advanced techniques

---

**Questions?** Open an issue at https://github.com/anthropics/claude-code/issues
