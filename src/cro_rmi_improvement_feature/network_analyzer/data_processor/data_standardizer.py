import os
import json
import pandas as pd
from typing import Union, List, Dict, Any
import warnings
from pathlib import Path

from torch import set_anomaly_enabled

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def determine_risk_level(row: pd.Series, risk_type: str = "combined") -> int:
    """
    Determine risk level based on risk score and impact.

    Args:
        row: DataFrame row containing risk score and impact data
        risk_type: Type of risk calculation ("fin", "nonfin", or "combined")

    Returns:
        Risk level (0-4)
    """
    if risk_type == "fin":
        riskscore = "risk_score_fin"
        riskimpact = "impact_fin_combined"
    elif risk_type == "nonfin":
        riskscore = "risk_score_nonfin"
        riskimpact = "impact_nonfin_combined"
    else:
        riskscore = "risk_score"
        riskimpact = "impact_combined"

    if row[riskscore] >= 20:
        return 4
    elif 10 <= row[riskscore] <= 16:
        return 3
    elif 4 <= row[riskscore] <= 9:
        return 2
    elif (0 < row[riskscore] < 4) or (row[riskscore] == 4 and row[riskimpact] == 2):
        return 1
    elif row[riskscore] == 0:
        return 0
    else:
        return 0


def list_string_to_list(list_string: str) -> tuple[bool, Union[list, str]]:
    """
    Convert string representation of list to actual list.

    Args:
        list_string: String that may represent a list

    Returns:
        Tuple of (is_list, result)
    """
    is_list = False
    if ((list_string.count("'") % 2 != 0) and (list_string.count("'") != 0)) or (
        (list_string.count('"') % 2 != 0) and (list_string.count('"') != 0)
    ):
        return is_list, list_string

    is_list = True
    # Swap quotes for JSON parsing
    list_string = list_string.replace('"', "###")
    list_string = list_string.replace("'", '"')
    list_string = list_string.replace("###", "'")

    try:
        list_cell = json.loads(list_string)
    except:
        is_list = False
        return is_list, list_string

    return is_list, list_cell


def cell_to_list(cell: Any) -> Union[List, Any]:
    """
    Convert cell content to list format, handling sets and complex structures.

    Args:
        cell: Cell content that may be a set or other data structure

    Returns:
        Processed cell content
    """
    empty_element_list = ["", "-"]

    if isinstance(cell, set):
        new_cell = []
        for list_string_candidate in cell:
            if list_string_candidate in empty_element_list:
                continue
            try:
                is_list, new_list_string_candidate = list_string_to_list(
                    list_string_candidate
                )
                if is_list is False:
                    new_cell.append(new_list_string_candidate)
                    continue
                for i in new_list_string_candidate:
                    if i in empty_element_list:
                        continue
                    is_list, i = list_string_to_list(i)
                    if is_list is False:
                        new_cell.append(i)
                        continue
                    new_cell.extend(i)
            except:
                new_cell.append(list_string_candidate)
        return new_cell
    return cell


def merge_rootcause(row: pd.Series) -> str:
    """
    Merge rootcause and rootcause_desc columns.

    Args:
        row: DataFrame row containing rootcause and rootcause_desc

    Returns:
        Merged rootcause string
    """
    rootcause = row["rootcause"]
    rootcause_desc = row["rootcause_desc"]

    if rootcause == "" and rootcause_desc == "":
        return ""
    elif rootcause == "":
        return "rootcause_desc :" + rootcause_desc
    elif rootcause_desc == "":
        return "rootcause :" + rootcause
    else:
        return "rootcause :" + rootcause + "\n" + "rootcause_desc :" + rootcause_desc


def merge_process(row: pd.Series) -> str:
    """
    Merge process and process_desc columns.

    Args:
        row: DataFrame row containing process and process_desc

    Returns:
        Merged process string
    """
    process = row["process"]
    process_desc = row["process_desc"]

    if process == "" and process_desc == "":
        return ""
    elif process == "":
        return "process_desc :" + process_desc
    elif process_desc == "":
        return "process :" + process
    else:
        return "process :" + process + "\n" + "process_desc :" + process_desc


def preprocess_pcg_data(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    """
    Preprocess data specifically for PCG company.

    Args:
        df: Input DataFrame
        company_name: Name of the company

    Returns:
        Preprocessed DataFrame
    """
    # Add company column
    df["company"] = company_name
    df["process_desc"] = ""
    df["rootcause_desc"] = ""

    # Calculate risk score
    df["risk_score"] = df["likelihood_combined"] * df["impact_combined"]

    # Determine risk level
    df["risk_level"] = df.apply(determine_risk_level, risk_type="combined", axis=1)

    # Rename columns
    df.rename(
        columns={
            "Risk Category": "risk_cat",
            "Risk Item": "risk",
            "Risk Description": "risk_desc",
            "Root Cause": "rootcause",
            "Process": "process",
            "Control": "control_name",
            "Control Description": "control_desc",
            "EC-Root Cause": "control_rootcause",
            "EC-Process": "control_process",
            "Design Score": "control_design_score",
            "Effective Score": "control_effective_score",
        },
        inplace=True,
    )

    return df


def preprocess_lotus_south_data(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    """
    Preprocess Lotus South data.
    """
    df["company"] = company_name
    df["process_desc"] = ""
    df["rootcause_desc"] = ""

    # Calculate risk score
    df["risk_score"] = df["likelihood_combined"] * df["impact_combined"]

    # Determine risk level
    df["risk_level"] = df.apply(determine_risk_level, risk_type="combined", axis=1)

    # Rename columns
    df.rename(
        columns={
            "Risk Category": "risk_cat",
            "Risk Item": "risk",
            "Risk Description": "risk_desc",
            "Root Cause": "rootcause",
            "Process": "process",
            "Control": "control_name",
            "Control Description": "control_desc",
            "EC-Root Cause": "control_rootcause",
            "EC-Process": "control_process",
            "Design Score": "control_design_score",
            "Effective Score": "control_effective_score",
        },
        inplace=True,
    )

    return df


def aggregate_risk_data(df: pd.DataFrame, score_method: str = "RMI") -> pd.DataFrame:
    """
    Aggregate risk data by company and risk.

    Args:
        df: Input DataFrame
        score_method: Method for scoring ("MAX" or "RMI")

    Returns:
        Aggregated DataFrame
    """
    if score_method == "MAX":
        df2 = df.groupby(["company", "risk"], as_index=False).agg(
            {
                "risk_desc": set,
                "risk_cat": "first",
                "rootcause": set,
                "rootcause_desc": set,
                "process": set,
                "process_desc": set,
                "risk_level": max,
                "likelihood_combined": max,
                "impact_combined": max,
                "control_name": lambda x: list(x),
                "control_desc": lambda x: list(x),
                "control_rootcause": lambda x: list(x),
                "control_process": lambda x: list(x),
                "control_design_score": lambda x: list(x),
                "control_effective_score": lambda x: list(x),
            }
        )
        df2["risk_score"] = df2["impact_combined"] * df2["likelihood_combined"]
        df2["risk_level"] = df2.apply(
            determine_risk_level, risk_type="combined", axis=1
        )

    elif score_method == "RMI":
        df2 = df.copy()
        df2["risk_score"] = df2["impact_combined"] * df2["likelihood_combined"]
        df2_sorted = df2.sort_values(
            by=["company", "risk", "risk_score", "impact_combined"],
            ascending=[True, True, False, False],
        )

        df2 = df2_sorted.groupby(["company", "risk"], as_index=False).agg(
            {
                "risk_desc": lambda x: list(x),
                "risk_cat": "first",
                "impact_combined": "first",
                "risk_score": "max",
                "risk_level": "max",
                "rootcause": lambda x: list(x),
                "rootcause_desc": lambda x: list(x),
                "process": lambda x: list(x),
                "process_desc": lambda x: list(x),
                "control_name": lambda x: list(x),
                "control_desc": lambda x: list(x),
                "control_rootcause": lambda x: list(x),
                "control_process": lambda x: list(x),
                "control_design_score": lambda x: list(x),
                "control_effective_score": lambda x: list(x),
            }
        )

        df2["likelihood_combined"] = df2["risk_score"] / df2["impact_combined"]
        df2["risk_level"] = df2.apply(
            determine_risk_level, risk_type="combined", axis=1
        )

    else:
        raise ValueError(f"Invalid score method: {score_method}")

    return df2


def process_catalog_data(
    catalog_path: str, date_stamp: str, selected_risk_cat: List[str]
) -> pd.DataFrame:
    """
    Process risk catalog data.

    Args:
        catalog_path: Path to catalog Excel file
        date_stamp: Date stamp for naming
        selected_risk_cat: List of selected risk categories

    Returns:
        Processed catalog DataFrame
    """
    risk_df = pd.read_excel(catalog_path, sheet_name="Risk_Translate")
    risk_cause_mapping_df = pd.read_excel(catalog_path, sheet_name="Risk_Cause_mapping")

    # Fix category naming
    risk_df["Risk-category"] = risk_df["Risk-category"].replace(
        "Operational risk", "Operational Risk"
    )

    # Add root causes from mapping
    for idx, row in risk_df.iterrows():
        risk_en = row["Risk-EN"]
        risk_cause_mapping_df_row = risk_cause_mapping_df[
            risk_cause_mapping_df["RiskName"] == risk_en
        ]
        risk_cause_list = risk_cause_mapping_df_row["RiskCause"].tolist()
        risk_cause_str = "\n".join(["- " + i for i in risk_cause_list])
        risk_df.loc[idx, "Root cause-EN"] = risk_cause_str

    # Create risk description mapping
    selected_cols = ["Risk-EN", "Description-EN"]
    risk_and_risk_desc_df = risk_df[selected_cols]
    risk_and_risk_desc_df = risk_and_risk_desc_df.map(
        lambda x: x.strip() if isinstance(x, str) else x
    )

    risk_to_desc_dict = risk_and_risk_desc_df.set_index("Risk-EN")[
        "Description-EN"
    ].to_dict()

    # Create new rows for catalog data
    new_rows = []
    for index, row in risk_df.iterrows():
        risk_cat = row["Risk-category"]
        if risk_cat not in selected_risk_cat:
            continue
        risk_desc = row["Description-EN"]
        risk_name = row["Risk-EN"]
        risk_rootcause = row["Root cause-EN"]
        data = {
            "company": f"risk_catalog-{date_stamp}",
            "risk_cat": risk_cat,
            "risk": risk_name,
            "risk_desc": risk_desc,
            "rootcause": risk_rootcause,
            "process": "",
            "risk_level": 0,
            "risk_score": 0,
            "impact_combined": 0,
            "likelihood_combined": 0,
            "control_name": [],
            "control_desc": [],
            "control_rootcause": [],
            "control_process": [],
            "control_design_score": [],
            "control_effective_score": [],
        }
        new_rows.append(data)

    return pd.DataFrame(new_rows), risk_to_desc_dict


def process_risk_data(
    risk_data_path: str,
    catalog_data_path: str,
    company_name: str,
    date_stamp: str,
    output_path: str = None,
) -> List[Dict[str, Any]]:
    """
    Main function to process risk data.

    Args:
        risk_data_path: Path to risk assessment Excel file
        catalog_data_path: Path to risk catalog Excel file
        company_name: Name of the company for preprocessing
        output_path: Optional path to save JSON output

    Returns:
        List of processed risk data dictionaries
    """
    # Load risk data
    ori_df = pd.read_excel(risk_data_path)

    # Preprocess based on company
    if company_name.upper() == "PCG":
        df = preprocess_pcg_data(ori_df, company_name)
    elif company_name.upper() == "LOTUS_SOUTH":
        df = preprocess_lotus_south_data(ori_df, company_name)
    else:
        raise ValueError(f"Invalid company name: {company_name}")

    # Define selected risk categories
    selected_risk_cat = [
        "Operational Risk",
        "Strategic Risk",
        "Credit Risk",
        "Market Risk",
        "Liquidity Risk",
    ]

    # Aggregate data
    df2 = aggregate_risk_data(df, score_method="RMI")
    print(f"df2 columns: {list(df2.columns)}")

    # Filter by company
    df3 = df2[df2["company"].isin([company_name])]
    print(f"df3 columns: {list(df3.columns)}")

    # Select relevant columns
    base_columns = [
        "company",
        "risk_cat",
        "risk",
        "risk_desc",
        "rootcause",
        "rootcause_desc",
        "process",
        "process_desc",
        "risk_level",
        "risk_score",
        "impact_combined",
        "likelihood_combined",
    ]

    # Add control columns if they exist
    control_columns = [
        "control_name",
        "control_desc",
        "control_rootcause",
        "control_process",
        "control_design_score",
        "control_effective_score",
    ]

    # Only include control columns that exist in the DataFrame
    available_columns = [
        col for col in base_columns + control_columns if col in df3.columns
    ]
    print(f"df3 available columns: {available_columns}")
    df4 = df3[available_columns]
    print(f"df4 columns: {list(df4.columns)}")

    # Convert sets to lists for specific columns that need it
    set_columns = [
        "risk_desc",
        "rootcause",
        "rootcause_desc",
        "process",
        "process_desc",
    ]
    control_columns_present = [
        col
        for col in [
            "control_name",
            "control_desc",
            "control_rootcause",
            "control_process",
            "control_design_score",
            "control_effective_score",
        ]
        if col in df4.columns
    ]

    df5 = df4.copy()
    for col in set_columns + control_columns_present:
        if col in df5.columns:
            df5[col] = df5[col].apply(cell_to_list)

    print(f"df5 columns: {list(df5.columns)}")

    # Convert lists to strings, handling mixed types
    def safe_join(x):
        if isinstance(x, list):
            # Convert all items to strings before joining
            return ",".join(str(item) for item in x)
        return x

    df6 = df5.apply(safe_join)
    print(f"df6 columns: {list(df6.columns)}")

    # Keep rootcause and process columns separate (no merging)
    df7 = df6.copy()
    print(f"df7 columns: {list(df7.columns)}")

    # Process catalog data
    catalog_df, risk_to_desc_dict = process_catalog_data(
        catalog_data_path, date_stamp, selected_risk_cat
    )
    print(f"catalog_df columns: {list(catalog_df.columns)}")

    # Merge with catalog descriptions
    df8 = df7.copy()
    df8["risk"] = df8["risk"].str.strip()
    df8["risk_desc_catalog"] = df8["risk"].map(risk_to_desc_dict)
    df8["risk_desc_catalog"] = df8["risk_desc_catalog"].fillna("")
    df8["risk_desc_catalog"] = df8["risk_desc_catalog"].str.strip()

    # Combine descriptions - handle potential list values
    def safe_concat(row):
        catalog_desc = row["risk_desc_catalog"]
        risk_desc = row["risk_desc"]

        # Convert to strings if they are lists
        if isinstance(catalog_desc, list):
            catalog_desc = ",".join(str(item) for item in catalog_desc)
        if isinstance(risk_desc, list):
            risk_desc = ",".join(str(item) for item in risk_desc)

        # Handle None/NaN values
        if pd.isna(catalog_desc) or catalog_desc == "":
            return str(risk_desc) if not pd.isna(risk_desc) else ""
        if pd.isna(risk_desc) or risk_desc == "":
            return str(catalog_desc) if not pd.isna(catalog_desc) else ""

        return str(catalog_desc) + " " + str(risk_desc)

    df8["risk_desc"] = df8.apply(safe_concat, axis=1)
    df8["risk_desc"] = df8["risk_desc"].str.strip()
    df8.drop("risk_desc_catalog", axis=1, inplace=True)

    # Add catalog data
    df8 = pd.concat([df8, catalog_df], ignore_index=True)
    print(f"df8 columns: {list(df8.columns)}")

    # Convert to list of dictionaries
    result_list = df8.to_dict(orient="records")

    # Add date_stamp to each record
    for record in result_list:
        record["date_stamp"] = date_stamp

    # Save to JSON if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Replace NaN values, empty strings, and empty lists with None (null in JSON) before saving
        def replace_nan(obj):
            if isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                if len(obj) == 0:
                    return None
                return [replace_nan(item) for item in obj]
            elif pd.isna(obj):
                return None
            elif isinstance(obj, str) and obj.strip() == "":
                return None
            else:
                return obj

        cleaned_result_list = replace_nan(result_list)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_result_list, f, ensure_ascii=False, indent=2)

    return result_list


def load_standardized_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all standardized JSON files from the data directory.

    Args:
        data_dir: Path to the directory containing standardized JSON files

    Returns:
        List of all risk data records from all files
    """
    all_data = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all JSON files in the directory
    json_files = list(data_path.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    for json_file in json_files:
        print(f"Loading data from: {json_file}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                all_data.extend(file_data)
                print(f"Loaded {len(file_data)} records from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    print(f"Total records loaded: {len(all_data)}")
    return all_data


def merge_all_data(
    standardized_data: List[Dict[str, Any]],
    remove_duplicates: bool = True,
    duplicate_strategy: str = "keep_first",
) -> pd.DataFrame:
    """
    Merge all standardized data into a single DataFrame with optional duplicate removal.

    Args:
        standardized_data: List of risk data records
        remove_duplicates: Whether to remove duplicate records
        duplicate_strategy: Strategy for handling duplicates ("keep_first", "keep_last", "drop_all")

    Returns:
        Merged DataFrame with all data
    """
    if not standardized_data:
        raise ValueError("No data to merge")

    # Convert to DataFrame
    df = pd.DataFrame(standardized_data)

    # Ensure all records have required fields
    required_fields = ["company", "risk", "risk_desc", "date_stamp"]
    missing_fields = [field for field in required_fields if field not in df.columns]

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    initial_count = len(df)
    print(f"Initial data shape: {df.shape}")

    if remove_duplicates:
        print(f"Removing duplicates using strategy: {duplicate_strategy}")

        # Create combined fields for rootcause and process data
        # Handle list fields by concatenating with descriptions
        if "rootcause" in df.columns or "rootcause_desc" in df.columns:

            def combine_rootcause_data(row):
                """Combine rootcause and rootcause_desc into a single field."""
                rootcause = row.get("rootcause", "")
                rootcause_desc = row.get("rootcause_desc", "")

                # Handle None values
                if rootcause is None:
                    rootcause = ""
                if rootcause_desc is None:
                    rootcause_desc = ""

                # Convert to string if it's a list
                if isinstance(rootcause, list):
                    rootcause = ", ".join(
                        [str(item) for item in rootcause if item is not None]
                    )
                else:
                    rootcause = str(rootcause)

                if isinstance(rootcause_desc, list):
                    rootcause_desc = ", ".join(
                        [str(item) for item in rootcause_desc if item is not None]
                    )
                else:
                    rootcause_desc = str(rootcause_desc)

                # Combine and clean up
                combined = f"{rootcause} {rootcause_desc}".strip()
                return combined if combined else ""

            df["rootcause_data"] = df.apply(combine_rootcause_data, axis=1)
            print("Created rootcause_data field")

        if "process" in df.columns or "process_desc" in df.columns:

            def combine_process_data(row):
                """Combine process and process_desc into a single field."""
                process = row.get("process", "")
                process_desc = row.get("process_desc", "")

                # Handle None values
                if process is None:
                    process = ""
                if process_desc is None:
                    process_desc = ""

                # Convert to string if it's a list
                if isinstance(process, list):
                    process = ", ".join(
                        [str(item) for item in process if item is not None]
                    )
                else:
                    process = str(process)

                if isinstance(process_desc, list):
                    process_desc = ", ".join(
                        [str(item) for item in process_desc if item is not None]
                    )
                else:
                    process_desc = str(process_desc)

                # Combine and clean up
                combined = f"{process} {process_desc}".strip()
                return combined if combined else ""

            df["process_data"] = df.apply(combine_process_data, axis=1)
            print("Created process_data field")

        # Define columns to check for duplicates
        # Use the new combined fields instead of separate rootcause/process fields
        duplicate_columns = ["company", "risk", "risk_desc"]

        # Add date_stamp if available for more precise duplicate detection
        if "date_stamp" in df.columns:
            duplicate_columns.append("date_stamp")

        # Add the new combined fields
        if "rootcause_data" in df.columns:
            duplicate_columns.append("rootcause_data")
        if "process_data" in df.columns:
            duplicate_columns.append("process_data")

        # Add other relevant fields if they exist (but not the original list fields)
        optional_columns = ["risk_desc_catalog"]
        for col in optional_columns:
            if col in df.columns:
                duplicate_columns.append(col)

        # Filter out columns that contain unhashable types (like lists)
        safe_duplicate_columns = []
        for col in duplicate_columns:
            if col in df.columns:
                # Check if the column contains any list values
                sample_values = (
                    df[col].dropna().head(100)
                )  # Check first 100 non-null values
                has_lists = any(isinstance(val, list) for val in sample_values)

                if not has_lists:
                    safe_duplicate_columns.append(col)
                else:
                    print(
                        f"Warning: Column '{col}' contains list values, skipping from duplicate detection"
                    )

        print(f"Checking duplicates based on columns: {safe_duplicate_columns}")

        if not safe_duplicate_columns:
            print(
                "Warning: No safe columns for duplicate detection. Skipping duplicate removal."
            )
        else:
            # Count duplicates before removal
            duplicate_count = df.duplicated(subset=safe_duplicate_columns).sum()
            print(f"Found {duplicate_count} duplicate records")

            if duplicate_count > 0:
                if duplicate_strategy == "keep_first":
                    df = df.drop_duplicates(subset=safe_duplicate_columns, keep="first")
                elif duplicate_strategy == "keep_last":
                    df = df.drop_duplicates(subset=safe_duplicate_columns, keep="last")
                elif duplicate_strategy == "drop_all":
                    # Keep only records that appear exactly once
                    df = df.drop_duplicates(subset=safe_duplicate_columns, keep=False)
                else:
                    raise ValueError(
                        f"Invalid duplicate_strategy: {duplicate_strategy}"
                    )

                final_count = len(df)
                removed_count = initial_count - final_count
                print(f"Removed {removed_count} duplicate records")
                print(f"Final data shape: {df.shape}")
            else:
                print("No duplicates found")
    else:
        print("Duplicate removal skipped")

    print(f"Companies in data: {df['company'].unique()}")
    print(f"Date stamps in data: {df['date_stamp'].unique()}")

    return df


def save_merged_data(df: pd.DataFrame, output_path: str):
    """
    Save merged DataFrame to JSON format.

    Args:
        df: DataFrame with merged data (without embeddings)
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert DataFrame to list of dicts for JSON serialization
    data_to_save = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    print(f"Merged data saved to: {output_path}")


def process_merged_data(
    standardized_data_dir: str,
    processed_output_path: str,
    remove_duplicates: bool = True,
    duplicate_strategy: str = "keep_first",
) -> pd.DataFrame:
    """
    Process and merge all standardized data, saving to processed folder.

    Args:
        standardized_data_dir: Directory containing standardized JSON files
        processed_output_path: Path to save the merged data JSON file
        remove_duplicates: Whether to remove duplicate records
        duplicate_strategy: Strategy for handling duplicates ("keep_first", "keep_last", "drop_all")

    Returns:
        DataFrame with merged data (without embeddings)
    """
    print("Starting data merging process...")

    # Load all standardized data
    print("Loading standardized data...")
    standardized_data = load_standardized_data(standardized_data_dir)

    # Merge all data with duplicate removal
    print("Merging data...")
    merged_df = merge_all_data(
        standardized_data,
        remove_duplicates=remove_duplicates,
        duplicate_strategy=duplicate_strategy,
    )

    # Save merged data to processed folder
    print("Saving merged data...")
    save_merged_data(merged_df, processed_output_path)

    print("Data merging process completed!")
    return merged_df


def standardize_multiple_datasets():
    """
    Process multiple datasets for different companies and dates.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Define datasets to process
    datasets = [
        {
            "risk_data_path": os.path.join(
                dir_path,
                "../data/raw/",
                "250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx",
            ),
            "catalog_data_path": os.path.join(
                dir_path, "../data/raw/", "seed_excel_13May2025 1.xlsx"
            ),
            "company_name": "PCG",
            "date_stamp": "20250513",
            "output_path": os.path.join(
                dir_path, "../data/standardized/", "20250513-PCG_risk_data.json"
            ),
        },
        {
            "risk_data_path": os.path.join(
                dir_path,
                "../data/raw/",
                "250520_Retail (Guangdong-Guangxi)_assessment_report_Q1-2025_controlperrow_original.xlsx",
            ),
            "catalog_data_path": os.path.join(
                dir_path, "../data/raw/", "seed_excel_13May2025 1.xlsx"
            ),
            "company_name": "lotus_south",
            "date_stamp": "20250513",
            "output_path": os.path.join(
                dir_path, "../data/standardized/", "20250513-lotus_south_risk_data.json"
            ),
        },
    ]

    for dataset in datasets:
        print(f"\nProcessing {dataset['company_name']} dataset...")
        try:
            result = process_risk_data(
                risk_data_path=dataset["risk_data_path"],
                catalog_data_path=dataset["catalog_data_path"],
                company_name=dataset["company_name"],
                output_path=dataset["output_path"],
                date_stamp=dataset["date_stamp"],
            )
            print(
                f"\tSuccessfully processed {len(result)} risk records for {dataset['company_name']}"
            )
            print(f"Output saved to: {dataset['output_path']}")
        except Exception as e:
            print(f"Error processing {dataset['company_name']} data: {e}")


# Flag to control sample run
IS_SAMPLE_RUN = True


def main():
    """
    Example usage of the data standardizer and merger.
    """
    # Get the directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Define paths
    standardized_data_dir = os.path.join(dir_path, "../data/standardized/")
    processed_output_path = os.path.join(
        dir_path, "../data/processed/", "riskview_merged_data.json"
    )

    try:
        # Standardize multiple datasets
        standardize_multiple_datasets()

        # Process and merge all standardized data
        print("Starting data standardization and merging...")
        merged_df = process_merged_data(
            standardized_data_dir,
            processed_output_path,
            remove_duplicates=True,  # Enable duplicate removal
            duplicate_strategy="keep_first",  # Keep the first occurrence of duplicates
        )

        print(f"Processing completed! Processed {len(merged_df)} records")
        company_list = merged_df["company"].unique()
        print("\n")

        for company in company_list:
            print(
                f"\tCompany: {company} number of records: {len(merged_df[merged_df['company'] == company])}"
            )
            print("\n")
        print(f"Merged data saved to: {processed_output_path}")
        print(f"Duplicate removal: Enabled (strategy: keep_first)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure standardized data exists in ../data/standardized/")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
