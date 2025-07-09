import os
import pandas as pd
from granularity_classifier import classify_granularity, RISK_CATEGORIES
from granularity_generalizer import granularity_generalizer
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Any, Optional


def load_data(
    data_path: str,
    sample_run: bool = False,
    sample_filter: str = "business interruption",
) -> pd.DataFrame:
    """
    Load and optionally filter the dataset.

    Args:
        data_path: Path to the Excel file
        sample_run: Whether to run on a sample of the data
        sample_filter: Filter criteria for sample run

    Returns:
        Filtered DataFrame
    """
    df = pd.read_excel(data_path)
    print(f"Original dataset: {len(df)} risks")
    # filter to Operation column value = "Operation"
    df = df[df["Operation"] == "Operation"]
    if sample_run:
        # Filter for risks that start with the sample filter (case-insensitive)
        sample_mask = df["Risk-EN"].str.lower().str.startswith(sample_filter.lower())
        filtered_df = df[sample_mask].copy()

        print(f"Sample dataset ({sample_filter}): {len(filtered_df)} risks")

        if len(filtered_df) == 0:
            raise ValueError(f"No risks found starting with '{sample_filter}'.")

        df = filtered_df

    return df


def run_classifier(df: pd.DataFrame, dry_run: bool = False) -> List[Dict[str, Any]]:
    """
    Run granularity classification on the dataset.

    Args:
        df: DataFrame containing risk data
        dry_run: Whether to run only on first item for testing

    Returns:
        List of classification results
    """
    risk_name_en_list = df["Risk-EN"].tolist()
    risk_desc_en_list = df["Description-EN"].tolist()

    print(f"Processing {len(risk_name_en_list)} risks for classification...")

    # Test run on first item
    print("Testing classification on first item...")
    test_result = classify_granularity(
        text=risk_name_en_list[0],
        description=risk_desc_en_list[0],
        context="risk",
        reference_categories=RISK_CATEGORIES,
    )
    print(f"Test result: {test_result}")

    if dry_run:
        print("Dry run completed. Exiting.")
        return [test_result]

    # Process all items
    results = [test_result]  # Include the test result
    for i in tqdm(
        range(1, len(risk_name_en_list)), desc="Processing risk classifications"
    ):
        result = classify_granularity(
            text=risk_name_en_list[i],
            description=risk_desc_en_list[i],
            context="risk",
            reference_categories=RISK_CATEGORIES,
        )
        results.append(result)

    return results


def run_generalizer(
    df: pd.DataFrame,
    classification_results: List[Dict[str, Any]],
    context: str = "risk",
    reference_categories: Optional[List[str]] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run granularity generalizer on specific and too specific data only.

    Args:
        df: DataFrame containing risk data
        classification_results: Results from the classifier
        context: The domain context ("risk", "control", "rootcause", "process")
        reference_categories: Optional list of reference categories for "too general" classification
        dry_run: Whether to run only on first item for testing

    Returns:
        List of generalization results
    """
    risk_name_en_list = df["Risk-EN"].tolist()
    risk_desc_en_list = df["Description-EN"].tolist()

    # Filter for specific and too specific items
    items_to_generalize = []
    for i, result in enumerate(classification_results):
        if result["level"] in ["specific", "too specific"]:
            items_to_generalize.append(
                {
                    "index": i,
                    "text": risk_name_en_list[i],
                    "description": risk_desc_en_list[i],
                    "granularity": result["level"],
                }
            )

    print(
        f"Found {len(items_to_generalize)} items to generalize (specific/too specific)"
    )

    if not items_to_generalize:
        print("No items to generalize. Exiting.")
        return []

    # Test run on first item to generalize
    print("Testing generalization on first specific/too specific item...")
    test_item = items_to_generalize[0]
    test_result = granularity_generalizer(
        text=test_item["text"],
        description=test_item["description"],
        granularity=test_item["granularity"],
        context=context,
        reference_categories=reference_categories,
        model_name="gpt-4.1-mini",
    )
    print(f"Test generalization result: {test_result}")

    if dry_run:
        print("Dry run completed. Exiting.")
        return [{"index": test_item["index"], "result": test_result}]

    # Process all items to generalize
    generalization_results = [{"index": test_item["index"], "result": test_result}]

    for item in tqdm(items_to_generalize[1:], desc="Processing generalizations"):
        result = granularity_generalizer(
            text=item["text"],
            description=item["description"],
            granularity=item["granularity"],
            context=context,
            reference_categories=reference_categories,
            model_name="gpt-4.1-mini",
        )
        generalization_results.append({"index": item["index"], "result": result})

    return generalization_results


def save_results(
    df: pd.DataFrame,
    classification_results: List[Dict[str, Any]],
    generalization_results: List[Dict[str, Any]],
    results_dir: str,
):
    """
    Save classification and generalization results to Excel.

    Args:
        df: Original DataFrame
        classification_results: Results from classifier
        generalization_results: Results from generalizer
        results_dir: Directory to save results
    """
    # Create results folder if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Prepare classification data for Excel export
    classification_data = []
    for i, result in enumerate(classification_results):
        classification_data.append(
            {
                "Risk-EN": df.iloc[i]["Risk-EN"],
                "Description-EN": df.iloc[i]["Description-EN"],
                "Granularity_Level": result["level"],
                "Confidence": result.get("confidence", "N/A"),
                "Reasoning": result.get("reasoning", "N/A"),
            }
        )

    # Create generalization data for Excel export
    generalization_data = []
    for gen_result in generalization_results:
        index = gen_result["index"]
        result = gen_result["result"]
        generalization_data.append(
            {
                "Risk-EN": df.iloc[index]["Risk-EN"],
                "Description-EN": df.iloc[index]["Description-EN"],
                "Original_Granularity": classification_results[index]["level"],
                "Too_Specific_Version": result.get("too_specific"),
                "Specific_Version": result.get("specific"),
                "General_Version": result.get("general"),
                "Too_General_Version": result.get("too_general"),
            }
        )

    # Save classification results
    classification_df = pd.DataFrame(classification_data)
    classification_path = os.path.join(
        results_dir, "granularity_classification_results.xlsx"
    )
    classification_df.to_excel(classification_path, index=False)
    print(f"Classification results saved to: {classification_path}")

    # Save generalization results
    if generalization_data:
        generalization_df = pd.DataFrame(generalization_data)
        generalization_path = os.path.join(
            results_dir, "granularity_generalization_results.xlsx"
        )
        generalization_df.to_excel(generalization_path, index=False)
        print(f"Generalization results saved to: {generalization_path}")

    # Print statistics
    level_counter = Counter([result["level"] for result in classification_results])
    print(f"\nTotal records processed: {len(classification_results)}")
    print("Level distribution:")
    for level, count in level_counter.items():
        print(f"  {level}: {count}")

    if generalization_results:
        print(f"\nGeneralization statistics:")
        print(f"  Items generalized: {len(generalization_results)}")
        print(
            f"  Items skipped (general/too general): {len(classification_results) - len(generalization_results)}"
        )


def main():
    """Main function to orchestrate the classification and generalization process."""

    # Configuration
    DRY_RUN = False  # Set to True for testing with first item only
    SAMPLE_RUN = False  # Set to False for full run
    SAMPLE_FILTER = "business interruption"  # Can be changed to any filter criteria

    # Context and reference categories configuration
    CONTEXT = "risk"  # Can be "risk", "control", "rootcause", "process"
    REFERENCE_CATEGORIES = {
        "risk": [
            "Cybersecurity Risk",
            "Operational Risk",
            "Financial Risk",
            "Strategic Risk",
            "Reputational Risk",
        ],
        "control": [
            "Technical Controls",
            "Administrative Controls",
            "Physical Controls",
            "Detective Controls",
            "Preventive Controls",
        ],
        "rootcause": [
            "System Failures",
            "Resource Management",
            "Infrastructure Issues",
            "Human Factors",
            "Process Failures",
        ],
        "process": [
            "Financial Processes",
            "HR Processes",
            "IT Processes",
            "Operational Processes",
            "Compliance Processes",
        ],
    }

    # File paths
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, "../data", "seed_excel_13May25.xlsx")
    results_dir = os.path.join(dir_path, "../results")

    try:
        # Step 1: Load data
        print("=== STEP 1: LOADING DATA ===")
        df = load_data(data_path, SAMPLE_RUN, SAMPLE_FILTER)
        print(df.info())

        # Step 2: Run classifier
        print("\n=== STEP 2: RUNNING CLASSIFIER ===")
        classification_results = run_classifier(df, DRY_RUN)

        # Step 3: Run generalizer (only on specific and too specific data)
        print("\n=== STEP 3: RUNNING GENERALIZER ===")
        print(f"Context: {CONTEXT}")
        print(f"Reference categories: {REFERENCE_CATEGORIES[CONTEXT]}")

        generalization_results = run_generalizer(
            df,
            classification_results,
            context=CONTEXT,
            reference_categories=REFERENCE_CATEGORIES[CONTEXT],
            dry_run=DRY_RUN,
        )

        # Step 4: Save results
        print("\n=== STEP 4: SAVING RESULTS ===")
        save_results(df, classification_results, generalization_results, results_dir)

        print("\n=== PROCESS COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
