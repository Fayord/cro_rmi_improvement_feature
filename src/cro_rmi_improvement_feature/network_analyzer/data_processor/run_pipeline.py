#!/usr/bin/env python3
"""
Risk Network Data Processing Pipeline Runner

This script runs the complete data processing pipeline in the correct order.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_module(module_name: str, description: str) -> bool:
    """
    Run a Python module and return success status.

    Args:
        module_name: Name of the module to run
        description: Description of what the module does

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: python {module_name}.py")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, f"{module_name}.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
            text=True,
        )

        if result.returncode == 0:
            print(f"✅ {module_name}.py completed successfully")
            return True
        else:
            print(f"❌ {module_name}.py failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Error running {module_name}.py: {e}")
        return False


def main():
    """
    Run the complete data processing pipeline.
    """
    data_type = "riskview"

    print("🚀 Starting Risk Network Data Processing Pipeline")
    print(f"Data Type: {data_type}")
    print("=" * 60)

    # Define the pipeline steps
    pipeline_steps = [
        ("data_standardizer", "Data Standardization"),
        ("summarize_data", "Data Summarization (Optional)"),
        ("embedding_process", "Embedding Processing"),
        ("create_graph", "Graph Creation"),
        ("relation_classifier", "Relationship Classification"),
    ]

    # Track success/failure
    results = []

    # Run each step
    for module_name, description in pipeline_steps:
        success = run_module(module_name, description)
        results.append((module_name, success))

        if not success:
            print(f"\n⚠️  Pipeline stopped due to failure in {module_name}")
            print("Please check the error messages above and fix the issue.")
            print("You can run individual modules to debug specific steps.")
            break

    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")

    for module_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{module_name}: {status}")

    # Check if all steps completed
    all_success = all(success for _, success in results)

    if all_success:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"All data processing steps have been completed for {data_type} data.")
        print("Check the output directories for results:")
        print(f"  - ../data/processed/ ({data_type}_* files)")
        print(f"  - ../data/embeddings/ ({data_type}_* files)")
        print(f"  - ../data/graphs/ ({data_type}_* files)")
        print(f"  - ../data/relationships/ ({data_type}_* files)")
    else:
        print(f"\n⚠️  Pipeline completed with errors.")
        print("Some steps failed. Check the output above for details.")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
