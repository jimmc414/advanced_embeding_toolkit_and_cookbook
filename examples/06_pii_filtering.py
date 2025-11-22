#!/usr/bin/env python3
"""
Example 6: PII Filtering and Safety
Demonstrates automatic detection and redaction of personally identifiable information.
"""

from embkit.lib.safety.pii import pii_contains, pii_redact, pii_filter_results

def main():
    print("=" * 60)
    print("Example 6: PII Detection and Redaction")
    print("=" * 60)

    # Sample search results with potential PII
    search_results = [
        {"id": "doc_001", "snippet": "Contact our support team at support@example.com for help"},
        {"id": "doc_002", "snippet": "Call us at +1 (555) 123-4567 for immediate assistance"},
        {"id": "doc_003", "snippet": "Our office is located at 123 Main Street"},
        {"id": "doc_004", "snippet": "Employee SSN: 123-45-6789 for tax records"},
        {"id": "doc_005", "snippet": "General information about our services"}
    ]

    print("\n📋 Original Search Results:")
    for result in search_results:
        has_pii = pii_contains(result["snippet"])
        marker = "⚠️  PII!" if has_pii else "✓ Clean"
        print(f"  {result['id']}: {marker}")
        print(f"    {result['snippet']}")

    # Apply PII filtering
    print("\n🔒 Applying PII Detection and Redaction...")
    filtered_results = pii_filter_results(search_results, field="snippet")

    print("\n📋 Filtered Results (PII Redacted):")
    for result in filtered_results:
        print(f"  {result['id']}:")
        print(f"    {result['snippet']}")

    # Individual text redaction
    print("\n" + "=" * 60)
    print("Manual Redaction Example")
    print("=" * 60)

    sensitive_text = "Please email me at john.doe@company.com or call 212-555-0100"
    print(f"\nOriginal: {sensitive_text}")
    print(f"Contains PII: {pii_contains(sensitive_text)}")
    print(f"Redacted: {pii_redact(sensitive_text)}")

    print("\n✓ Example complete!")
    print("\nKey Concept: PII filtering protects privacy by:")
    print("  - Detecting emails, phone numbers, SSNs via regex patterns")
    print("  - Redacting sensitive information before display")
    print("  - Preventing accidental exposure of personal data")
    print("\nPatterns detected:")
    print("  - Email: name@domain.com")
    print("  - Phone: +1 (555) 123-4567, 212-555-0100")
    print("  - SSN: 123-45-6789")

if __name__ == "__main__":
    main()
