#!/usr/bin/env python3
"""
Example 13: Score Calibration with Temperature Scaling
Demonstrates calibrating confidence scores to match true probabilities.
"""

import numpy as np
from embkit.lib.calibrate.temperature import temperature_fit, temperature_apply
from embkit.lib.eval.metrics import expected_calibration_error

def main():
    print("=" * 60)
    print("Example 13: Score Calibration")
    print("=" * 60)

    # Simulate relevance prediction scenario
    # Logits: raw model scores (uncalibrated)
    # Labels: true relevance (1=relevant, 0=not relevant)
    np.random.seed(42)

    # Create synthetic data: model is overconfident
    n_samples = 100
    true_probs = np.random.beta(2, 2, n_samples)  # True underlying probabilities
    labels = (np.random.random(n_samples) < true_probs).astype(np.float32)

    # Simulate overconfident logits (too extreme)
    logits = np.zeros(n_samples, dtype=np.float32)
    for i, (label, prob) in enumerate(zip(labels, true_probs)):
        # Add overconfidence: scale up the logits
        if label == 1:
            logits[i] = np.random.normal(2.5, 0.5)  # High positive
        else:
            logits[i] = np.random.normal(-2.5, 0.5)  # High negative

    print(f"\n📊 Dataset: {n_samples} relevance judgments")
    print(f"   Positive: {int(labels.sum())}, Negative: {int((1-labels).sum())}")

    # Raw probabilities (uncalibrated)
    raw_probs = 1 / (1 + np.exp(-logits))

    # Compute calibration error before
    ece_before = expected_calibration_error(labels, raw_probs, n_bins=10)
    print(f"\n❌ Before Calibration:")
    print(f"   Expected Calibration Error (ECE): {ece_before:.4f}")

    # Show some examples
    print(f"\n   Sample predictions (uncalibrated):")
    for i in range(5):
        print(f"   Sample {i}: prob={raw_probs[i]:.3f}, actual={int(labels[i])}")

    # Fit temperature
    print(f"\n🔧 Fitting temperature parameter...")
    T = temperature_fit(labels, logits)
    print(f"   Optimal temperature: T = {T:.3f}")

    if T > 1:
        print(f"   → Model is overconfident (T>1 smooths predictions)")
    elif T < 1:
        print(f"   → Model is underconfident (T<1 sharpens predictions)")

    # Apply calibration
    calibrated_probs = temperature_apply(logits, T)

    # Compute calibration error after
    ece_after = expected_calibration_error(labels, calibrated_probs, n_bins=10)
    print(f"\n✓ After Calibration:")
    print(f"   Expected Calibration Error (ECE): {ece_after:.4f}")
    print(f"   Improvement: {(ece_before - ece_after)/ece_before * 100:.1f}%")

    # Show same examples
    print(f"\n   Sample predictions (calibrated):")
    for i in range(5):
        print(f"   Sample {i}: prob={calibrated_probs[i]:.3f}, actual={int(labels[i])}")

    # Binned analysis
    print(f"\n📈 Calibration Analysis (10 bins):")
    bins = np.linspace(0, 1, 11)
    print(f"   {'Bin Range':20s} {'Confidence':>12s} {'Accuracy':>12s} {'Gap':>8s}")
    print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*8}")

    for i in range(10):
        mask = (calibrated_probs >= bins[i]) & (calibrated_probs < bins[i+1])
        if mask.sum() > 0:
            avg_conf = calibrated_probs[mask].mean()
            avg_acc = labels[mask].mean()
            gap = abs(avg_conf - avg_acc)
            print(f"   [{bins[i]:.1f}, {bins[i+1]:.1f})       "
                  f"{avg_conf:12.3f} {avg_acc:12.3f} {gap:8.3f}")

    print("\n✓ Example complete!")
    print("\nKey Concept: Temperature scaling calibrates probability estimates:")
    print("  p_calibrated = sigmoid(logits / T)")
    print("  - T is fit to minimize negative log-likelihood")
    print("  - Preserves ranking but improves probability estimates")
    print("  - Essential for confidence-based decision thresholds")
    print("\nUse cases: relevance prediction, ranking confidence, uncertainty estimation")

if __name__ == "__main__":
    main()
