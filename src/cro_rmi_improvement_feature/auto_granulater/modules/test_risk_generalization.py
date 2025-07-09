#!/usr/bin/env python3
"""
Test script specifically for validating improved risk generalization.
This tests the examples provided by the user to ensure better general versions.
"""

from granularity_generalizer import granularity_generalizer


def test_risk_generalization_improvements():
    """Test the improved risk generalization with the specific examples provided."""

    # Test cases based on the user's examples
    test_cases = [
        {
            "name": "Business interruption from natural disasters",
            "text": "Business interruption from natural disasters",
            "description": "Risk of business disruption due to natural disasters",
            "granularity": "specific",
            "expected_general": "Business interruption",
            "expected_too_general": "Operational Risk",
        },
        {
            "name": "Business interruption from labor dispute",
            "text": "Business interruption from labor dispute",
            "description": "Risk of business disruption due to labor disputes",
            "granularity": "specific",
            "expected_general": "Business interruption",
            "expected_too_general": "Operational Risk",
        },
        {
            "name": "Business interruption from pandemic or epidemic",
            "text": "Business interruption from pandemic or epidemic",
            "description": "Risk of business disruption due to health crises",
            "granularity": "specific",
            "expected_general": "Business interruption",
            "expected_too_general": "Operational Risk",
        },
        {
            "name": "Business interruption from fire hazards",
            "text": "Business interruption from fire hazards",
            "description": "Risk of business disruption due to fire hazards",
            "granularity": "specific",
            "expected_general": "Business interruption",
            "expected_too_general": "Operational Risk",
        },
        {
            "name": "Business interruption from flood",
            "text": "Business interruption from flood",
            "description": "Risk of business disruption due to flooding",
            "granularity": "specific",
            "expected_general": "Business interruption",
            "expected_too_general": "Operational Risk",
        },
        {
            "name": "Business interruption from Terrorist/ War",
            "text": "Business interruption from Terrorist/ War",
            "description": "Risk of business disruption due to geopolitical events",
            "granularity": "specific",
            "expected_general": "Business interruption",
            "expected_too_general": "Operational Risk",
        },
        {
            "name": "Data breach from unauthorized access",
            "text": "Data breach from unauthorized access",
            "description": "Risk of data breach due to unauthorized access",
            "granularity": "specific",
            "expected_general": "Data breach",
            "expected_too_general": "Cybersecurity Risk",
        },
        {
            "name": "System failure from hardware malfunction",
            "text": "System failure from hardware malfunction",
            "description": "Risk of system failure due to hardware issues",
            "granularity": "specific",
            "expected_general": "System failure",
            "expected_too_general": "Operational Risk",
        },
    ]

    reference_categories = [
        "Cybersecurity Risk",
        "Operational Risk",
        "Financial Risk",
        "Strategic Risk",
        "Reputational Risk",
    ]

    print("=== TESTING IMPROVED RISK GENERALIZATION ===\n")

    passed_tests = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: {test_case['text']}")
        print(f"Expected General: {test_case['expected_general']}")
        print(f"Expected Too General: {test_case['expected_too_general']}")

        try:
            result = granularity_generalizer(
                text=test_case["text"],
                description=test_case["description"],
                granularity=test_case["granularity"],
                context="risk",
                reference_categories=reference_categories,
                model_name="gpt-4.1-mini",
            )

            general_version = result.get("general")
            too_general_version = result.get("too_general")

            print(f"Generated General: {general_version}")
            print(f"Generated Too General: {too_general_version}")

            # Validate general version
            general_passed = False
            if general_version:
                # Check if it matches expected or is appropriately abstract
                if general_version.lower() == test_case["expected_general"].lower():
                    general_passed = True
                elif (
                    "due to" not in general_version.lower()
                    and "from" not in general_version.lower()
                ):
                    # Check if it's appropriately abstract (no cause-specific details)
                    general_passed = True

            # Validate too general version
            too_general_passed = False
            if too_general_version:
                if too_general_version in reference_categories:
                    too_general_passed = True

            if general_passed and too_general_passed:
                print("✓ PASSED")
                passed_tests += 1
            else:
                print("✗ FAILED")
                if not general_passed:
                    print("  - General version issue: Should be more abstract")
                if not too_general_passed:
                    print(
                        "  - Too general version issue: Should use reference categories"
                    )

        except Exception as e:
            print(f"✗ ERROR: {str(e)}")

        print("-" * 60)

    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed_tests}/{total_tests} tests")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")


def test_common_pitfalls():
    """Test common pitfalls to ensure they're avoided."""

    print("\n=== TESTING COMMON PITFALLS ===\n")

    pitfalls = [
        {
            "name": "Should NOT include 'due to' in general version",
            "text": "Business interruption from natural disasters",
            "description": "Risk of business disruption",
            "granularity": "specific",
        },
        {
            "name": "Should NOT include 'from' in general version",
            "text": "Data breach from unauthorized access",
            "description": "Risk of data breach",
            "granularity": "specific",
        },
        {
            "name": "Should NOT include cause-specific details",
            "text": "System failure from hardware malfunction",
            "description": "Risk of system failure",
            "granularity": "specific",
        },
    ]

    reference_categories = ["Cybersecurity Risk", "Operational Risk", "Financial Risk"]

    for i, pitfall in enumerate(pitfalls, 1):
        print(f"Pitfall Test {i}: {pitfall['name']}")
        print(f"Input: {pitfall['text']}")

        try:
            result = granularity_generalizer(
                text=pitfall["text"],
                description=pitfall["description"],
                granularity=pitfall["granularity"],
                context="risk",
                reference_categories=reference_categories,
                model_name="gpt-4.1-mini",
            )

            general_version = result.get("general", "")

            # Check for common pitfalls
            has_due_to = "due to" in general_version.lower()
            has_from = "from" in general_version.lower()
            has_cause_specific = any(
                word in general_version.lower()
                for word in [
                    "natural",
                    "disaster",
                    "unauthorized",
                    "hardware",
                    "malfunction",
                ]
            )

            if not has_due_to and not has_from and not has_cause_specific:
                print(f"✓ PASSED - General version: {general_version}")
            else:
                print(f"✗ FAILED - General version: {general_version}")
                if has_due_to:
                    print("  - Contains 'due to' (should be removed)")
                if has_from:
                    print("  - Contains 'from' (should be removed)")
                if has_cause_specific:
                    print("  - Contains cause-specific details (should be removed)")

        except Exception as e:
            print(f"✗ ERROR: {str(e)}")

        print("-" * 40)


if __name__ == "__main__":
    # Run the main improvement tests
    test_risk_generalization_improvements()

    # Run the pitfall tests
    test_common_pitfalls()

    print("\nTesting completed!")
