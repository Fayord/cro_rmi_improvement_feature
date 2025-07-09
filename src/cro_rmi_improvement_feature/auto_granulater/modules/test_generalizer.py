#!/usr/bin/env python3
"""
Test script for the granularity_generalizer function.
This script demonstrates how the function works with different input granularity levels.
"""

from granularity_generalizer import granularity_generalizer


def test_generalizer():
    """Test the granularity generalizer with various examples."""

    # Test cases covering different scenarios
    test_cases = [
        {
            "name": "Too Specific to All Levels",
            "text": "Employee John Smith clicked on phishing email on March 15th, 2024 at 2:30 PM from sender phishing@malicious.com",
            "description": "Cybersecurity incident involving phishing attack targeting specific employee",
            "granularity": "too specific",
            "expected_generated": ["specific", "general", "too general"],
        },
        {
            "name": "Specific to General Levels",
            "text": "Firewall blocks unauthorized access attempts",
            "description": "Network security control measure to prevent unauthorized access",
            "granularity": "specific",
            "expected_generated": ["general", "too general"],
        },
        {
            "name": "General to Too General",
            "text": "Network security controls",
            "description": "Technical controls for network security",
            "granularity": "general",
            "expected_generated": ["too general"],
        },
        {
            "name": "Too General (No Generation)",
            "text": "Cybersecurity Risk",
            "description": "Broad risk category in cybersecurity domain",
            "granularity": "too general",
            "expected_generated": [],
        },
        {
            "name": "Business Process Example",
            "text": "User submitted expense report through online portal using Chrome browser",
            "description": "Financial process for expense reporting",
            "granularity": "specific",
            "expected_generated": ["general", "too general"],
        },
        {
            "name": "Root Cause Example",
            "text": "Database server crashed due to disk space exhaustion",
            "description": "System failure root cause analysis",
            "granularity": "specific",
            "expected_generated": ["general", "too general"],
        },
    ]

    print("=== GRANULARITY GENERALIZER TEST RESULTS ===\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Input Text: {test_case['text']}")
        print(f"Current Granularity: {test_case['granularity']}")
        print(f"Expected to generate: {test_case['expected_generated']}")

        try:
            # Call the generalizer function
            result = granularity_generalizer(
                text=test_case["text"],
                description=test_case["description"],
                granularity=test_case["granularity"],
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

            # Validate expectations
            print("\nValidation:")
            for expected_level in test_case["expected_generated"]:
                level_key = expected_level.replace(" ", "_")
                if result.get(level_key):
                    print(f"✓ {expected_level}: Generated as expected")
                else:
                    print(f"✗ {expected_level}: Expected but not generated")

            # Check that more specific levels are None
            hierarchy = {
                "too specific": 0,
                "specific": 1,
                "general": 2,
                "too general": 3,
            }
            input_level = hierarchy[test_case["granularity"]]

            for level, order in hierarchy.items():
                if order < input_level:  # More specific than input
                    level_key = level.replace(" ", "_")
                    if result.get(level_key) is None:
                        print(f"✓ {level}: Correctly None (cannot make more specific)")
                    else:
                        print(
                            f"✗ {level}: Should be None but got: {result.get(level_key)}"
                        )

        except Exception as e:
            print(f"Error: {str(e)}")

        print("\n" + "=" * 60 + "\n")


def test_edge_cases():
    """Test edge cases and error handling."""

    print("=== EDGE CASES TESTING ===\n")

    edge_cases = [
        {
            "name": "Empty Text",
            "text": "",
            "description": "Empty text input",
            "granularity": "specific",
        },
        {
            "name": "Very Short Text",
            "text": "Risk",
            "description": "Single word input",
            "granularity": "general",
        },
        {
            "name": "Very Long Text",
            "text": "A comprehensive cybersecurity incident occurred on March 15th, 2024 at 2:30 PM when employee John Smith, who works in the IT department on the 3rd floor of Building A, received a phishing email from sender phishing@malicious.com with subject line 'Urgent: Password Reset Required' and clicked on the embedded link, which led to a fake login page hosted on malicious-server.com, resulting in the compromise of his corporate credentials and subsequent unauthorized access to the company's internal systems including the customer database, financial records, and employee personal information, leading to a potential data breach affecting over 10,000 customers and 500 employees, with estimated financial impact of $2.5 million in remediation costs and potential regulatory fines.",
            "description": "Very detailed incident description",
            "granularity": "too specific",
        },
    ]

    for i, test_case in enumerate(edge_cases, 1):
        print(f"Edge Case {i}: {test_case['name']}")
        print(f"Text length: {len(test_case['text'])} characters")

        try:
            result = granularity_generalizer(
                text=test_case["text"],
                description=test_case["description"],
                granularity=test_case["granularity"],
            )

            print("Result: Success")
            for level, version in result.items():
                if version:
                    print(
                        f"  {level}: {version[:100]}{'...' if len(version) > 100 else ''}"
                    )

        except Exception as e:
            print(f"Result: Error - {str(e)}")

        print()


if __name__ == "__main__":
    # Run the main tests
    test_generalizer()

    # Run edge case tests
    test_edge_cases()

    print("Testing completed!")
