#!/usr/bin/env python3
"""
Test script for the updated granularity_generalizer function with context and reference categories.
"""

from granularity_generalizer import granularity_generalizer


def test_updated_generalizer():
    """Test the updated granularity generalizer with different contexts."""

    # Test cases for different contexts
    test_cases = [
        {
            "name": "Risk Context - Too Specific to All Levels",
            "text": "Employee John Smith clicked on phishing email on March 15th, 2024 at 2:30 PM from sender phishing@malicious.com",
            "description": "Cybersecurity incident involving phishing attack targeting specific employee",
            "granularity": "too specific",
            "context": "risk",
            "reference_categories": [
                "Cybersecurity Risk",
                "Operational Risk",
                "Financial Risk",
                "Strategic Risk",
                "Reputational Risk",
            ],
        },
        {
            "name": "Control Context - Specific to General Levels",
            "text": "Firewall blocks unauthorized access attempts on port 443",
            "description": "Network security control measure to prevent unauthorized access",
            "granularity": "specific",
            "context": "control",
            "reference_categories": [
                "Technical Controls",
                "Administrative Controls",
                "Physical Controls",
                "Detective Controls",
            ],
        },
        {
            "name": "Root Cause Context - Specific to General Levels",
            "text": "Database server crashed due to disk space exhaustion on /var/log partition",
            "description": "System failure root cause analysis",
            "granularity": "specific",
            "context": "rootcause",
            "reference_categories": [
                "System Failures",
                "Resource Management",
                "Infrastructure Issues",
                "Human Factors",
            ],
        },
        {
            "name": "Process Context - Specific to General Levels",
            "text": "User submitted expense report through online portal using Chrome browser",
            "description": "Financial process for expense reporting",
            "granularity": "specific",
            "context": "process",
            "reference_categories": [
                "Financial Processes",
                "HR Processes",
                "IT Processes",
                "Operational Processes",
            ],
        },
        {
            "name": "Risk Context - Too General (No Generation)",
            "text": "Cybersecurity Risk",
            "description": "Broad risk category in cybersecurity domain",
            "granularity": "too general",
            "context": "risk",
            "reference_categories": [
                "Cybersecurity Risk",
                "Operational Risk",
                "Financial Risk",
            ],
        },
    ]

    print("=== TESTING UPDATED GRANULARITY GENERALIZER ===\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Context: {test_case['context']}")
        print(f"Input Text: {test_case['text']}")
        print(f"Current Granularity: {test_case['granularity']}")
        print(f"Reference Categories: {test_case['reference_categories']}")

        try:
            # Call the updated generalizer function
            result = granularity_generalizer(
                text=test_case["text"],
                description=test_case["description"],
                granularity=test_case["granularity"],
                context=test_case["context"],
                reference_categories=test_case["reference_categories"],
                model_name="gpt-4.1-mini",
            )

            print("\nResults:")
            print("-" * 40)

            # Check each granularity level
            for level in ["too_specific", "specific", "general", "too_general"]:
                version = result.get(level)
                if version:
                    print(f"✓ {level}: {version}")
                else:
                    print(f"✗ {level}: None")

            # Validate context-specific behavior
            print("\nContext Validation:")
            if test_case["context"] == "risk":
                if result.get("too_general") in test_case["reference_categories"]:
                    print(
                        "✓ Too general version uses reference categories (risk context)"
                    )
                else:
                    print("✗ Too general version doesn't match reference categories")

        except Exception as e:
            print(f"Error: {str(e)}")

        print("\n" + "=" * 60 + "\n")


def test_context_validation():
    """Test that invalid contexts are properly handled."""

    print("=== TESTING CONTEXT VALIDATION ===\n")

    try:
        result = granularity_generalizer(
            text="Test text",
            description="Test description",
            granularity="specific",
            context="invalid_context",  # This should raise an error
            reference_categories=["Test Category"],
            model_name="gpt-4.1-mini",
        )
        print("✗ Should have raised an error for invalid context")
    except ValueError as e:
        print(f"✓ Correctly raised error for invalid context: {str(e)}")
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")


if __name__ == "__main__":
    # Run the main tests
    test_updated_generalizer()

    # Run context validation tests
    test_context_validation()

    print("Testing completed!")
