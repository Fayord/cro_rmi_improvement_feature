#!/usr/bin/env python3
"""
Example usage of the risk data processor module.

This script demonstrates how to use the risk_data_processor module
to process risk assessment data and generate JSON output.
"""

import os
import sys
from risk_data_processor import process_risk_data


def example_pcg_processing():
    """
    Example processing for PCG company data.
    """
    print("Processing PCG risk data...")

    # Define file paths
    risk_data_path = "data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx"
    catalog_data_path = "data/RMI-V2-Translate_20250508.xlsx"
    company_name = "PCG"
    output_path = "result/250528-pcg_risk_data.json"

    # Check if input files exist
    if not os.path.exists(risk_data_path):
        print(f"Error: Risk data file not found: {risk_data_path}")
        return

    if not os.path.exists(catalog_data_path):
        print(f"Error: Catalog data file not found: {catalog_data_path}")
        return

    try:
        # Process the data
        result = process_risk_data(
            risk_data_path=risk_data_path,
            catalog_data_path=catalog_data_path,
            company_name=company_name,
            output_path=output_path,
        )

        print(f"✅ Successfully processed {len(result)} risk records")
        print(f"📁 Output saved to: {output_path}")

        # Print first few records as example
        print("\n📋 Sample records:")
        for i, record in enumerate(result[:3]):
            print(f"\nRecord {i+1}:")
            print(f"  Company: {record['company']}")
            print(f"  Risk Category: {record['risk_cat']}")
            print(f"  Risk: {record['risk']}")
            print(f"  Risk Level: {record['risk_level']}")
            print(f"  Description: {record['risk_desc'][:100]}...")

    except Exception as e:
        print(f"❌ Error processing risk data: {e}")
        import traceback

        traceback.print_exc()


def example_generic_processing():
    """
    Example processing for generic company data.
    """
    print("\nProcessing generic company data...")

    # Define file paths (you would need to adjust these for your data)
    risk_data_path = "data/generic_risk_assessment.xlsx"
    catalog_data_path = "data/RMI-V2-Translate_20250508.xlsx"
    company_name = "GENERIC_COMPANY"
    output_path = "result/generic_risk_data.json"

    # Check if input files exist
    if not os.path.exists(risk_data_path):
        print(f"⚠️  Risk data file not found: {risk_data_path}")
        print(
            "   This is expected for the generic example - adjust the path for your data"
        )
        return

    try:
        # Process the data
        result = process_risk_data(
            risk_data_path=risk_data_path,
            catalog_data_path=catalog_data_path,
            company_name=company_name,
            output_path=output_path,
        )

        print(f"✅ Successfully processed {len(result)} risk records")
        print(f"📁 Output saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error processing risk data: {e}")


def main():
    """
    Main function to run examples.
    """
    print("🚀 Risk Data Processor - Example Usage")
    print("=" * 50)

    # Run PCG example
    example_pcg_processing()

    # Run generic example
    example_generic_processing()

    print("\n" + "=" * 50)
    print("✨ Example usage completed!")
    print("\nTo use this module in your own code:")
    print(
        """
from risk_data_processor import process_risk_data

result = process_risk_data(
    risk_data_path="path/to/your/risk_data.xlsx",
    catalog_data_path="path/to/your/catalog.xlsx", 
    company_name="YOUR_COMPANY",
    output_path="path/to/output.json"
)
    """
    )


if __name__ == "__main__":
    main()
