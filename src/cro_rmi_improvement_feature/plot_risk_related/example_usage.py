"""
Example Usage of RiskDataProcessor

This script demonstrates how to use the RiskDataProcessor class to process
risk assessment data from Excel files.

Usage:
    python example_usage.py
"""

import os
import sys
from risk_data_processor import RiskDataProcessor


def example_basic_usage():
    """Example of basic usage with default settings."""
    print("=== Basic Usage Example ===")

    # Initialize the processor
    processor = RiskDataProcessor()

    # Define file paths
    assessment_file = "data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx"
    catalog_file = "data/RMI-V2-Translate_20250508.xlsx"
    output_file = "result/250528-company_risk_data.json"

    # Process the data
    result = processor.process_complete_workflow(
        assessment_file=assessment_file,
        catalog_file=catalog_file,
        output_file=output_file,
        companies=["PCG"],
        aggregation_method="RMI",
    )

    print(f"Processed {len(result)} risk records")
    print(f"Output saved to: {output_file}")
    return result


def example_custom_risk_categories():
    """Example with custom risk categories."""
    print("\n=== Custom Risk Categories Example ===")

    # Define custom risk categories
    custom_categories = ["Operational Risk", "Strategic Risk", "Credit Risk"]

    # Initialize processor with custom categories
    processor = RiskDataProcessor(selected_risk_categories=custom_categories)

    # Define file paths
    assessment_file = "data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx"
    catalog_file = "data/RMI-V2-Translate_20250508.xlsx"
    output_file = "result/250528-custom_categories.json"

    # Process the data
    result = processor.process_complete_workflow(
        assessment_file=assessment_file,
        catalog_file=catalog_file,
        output_file=output_file,
        companies=["PCG"],
        aggregation_method="MAX",  # Using MAX method instead of RMI
    )

    print(f"Processed {len(result)} risk records with custom categories")
    return result


def example_step_by_step():
    """Example showing step-by-step processing."""
    print("\n=== Step-by-Step Processing Example ===")

    # Initialize processor
    processor = RiskDataProcessor()

    # Step 1: Load assessment data
    assessment_file = "data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx"
    df = processor.load_assessment_data(assessment_file, company_name="PCG")

    # Step 2: Calculate risk scores
    df = processor.calculate_risk_scores(df)

    # Step 3: Rename columns
    df = processor.rename_columns(df)

    # Step 4: Aggregate risk data
    df = processor.aggregate_risk_data(df, method="RMI")

    # Step 5: Filter companies
    df = processor.filter_companies(df, companies=["PCG"])

    # Step 6: Select columns
    df = processor.select_columns(df)

    # Step 7: Process data structures
    df = processor.process_data_structures(df)

    # Step 8: Merge rootcause and process columns
    df = processor.merge_rootcause_and_process(df)

    # Step 9: Load risk catalog
    catalog_file = "data/RMI-V2-Translate_20250508.xlsx"
    risk_catalog_df, risk_to_desc_dict = processor.load_risk_catalog(catalog_file)

    # Step 10: Add catalog risks
    date_stamp = processor.extract_date_stamp(assessment_file)
    df = processor.add_catalog_risks(df, risk_catalog_df, date_stamp)

    # Step 11: Convert to list and save
    result_list = df.to_dict(orient="records")
    output_file = "result/250528-step-by-step.json"
    processor.save_to_json(result_list, output_file)

    print(f"Step-by-step processing completed. Generated {len(result_list)} records.")
    return result_list


def example_data_analysis():
    """Example showing how to analyze the processed data."""
    print("\n=== Data Analysis Example ===")

    # Process the data first
    processor = RiskDataProcessor()
    assessment_file = "data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx"
    catalog_file = "data/RMI-V2-Translate_20250508.xlsx"

    # Load and process data
    df = processor.load_assessment_data(assessment_file)
    df = processor.calculate_risk_scores(df)
    df = processor.rename_columns(df)
    df = processor.aggregate_risk_data(df, method="RMI")
    df = processor.filter_companies(df, companies=["PCG"])
    df = processor.select_columns(df)
    df = processor.process_data_structures(df)
    df = processor.merge_rootcause_and_process(df)

    # Analyze the data
    print("Data Analysis Results:")
    print(f"Total number of risks: {len(df)}")
    print(f"Risk categories: {df['risk_cat'].unique()}")
    print(f"Risk level distribution:\n{df['risk_level'].value_counts().sort_index()}")

    # Find high-risk items (level 3 or 4)
    high_risk = df[df["risk_level"] >= 3]
    print(f"\nHigh-risk items (level 3-4): {len(high_risk)}")
    if len(high_risk) > 0:
        print("High-risk items:")
        for _, row in high_risk.iterrows():
            print(f"  - {row['risk']} (Level {row['risk_level']})")

    return df


def example_error_handling():
    """Example showing error handling."""
    print("\n=== Error Handling Example ===")

    processor = RiskDataProcessor()

    # Try to process with non-existent file
    try:
        result = processor.process_complete_workflow(
            assessment_file="non_existent_file.xlsx",
            catalog_file="data/RMI-V2-Translate_20250508.xlsx",
            output_file="result/error_test.json",
        )
    except FileNotFoundError as e:
        print(f"Expected error caught: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Try with invalid aggregation method
    try:
        result = processor.process_complete_workflow(
            assessment_file="data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx",
            catalog_file="data/RMI-V2-Translate_20250508.xlsx",
            output_file="result/error_test.json",
            aggregation_method="INVALID",
        )
    except ValueError as e:
        print(f"Expected error caught: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    """Run all examples."""
    print("Risk Data Processor - Example Usage")
    print("=" * 50)

    # Check if data files exist
    required_files = [
        "data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx",
        "data/RMI-V2-Translate_20250508.xlsx",
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Warning: The following required data files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure the data files are available before running examples.")
        return

    try:
        # Run examples
        example_basic_usage()
        example_custom_risk_categories()
        example_step_by_step()
        example_data_analysis()
        example_error_handling()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
