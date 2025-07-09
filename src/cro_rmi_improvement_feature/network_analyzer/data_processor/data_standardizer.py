import os
import json
import pandas as pd
from typing import Union, List, Dict, Any
import warnings

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


def process_multiple_datasets():
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
                dir_path, "../data/processed/", "20250513-PCG_risk_data.json"
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
                dir_path, "../data/processed/", "20250513-lotus_south_risk_data.json"
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
                f"Successfully processed {len(result)} risk records for {dataset['company_name']}"
            )
            print(f"Output saved to: {dataset['output_path']}")
        except Exception as e:
            print(f"Error processing {dataset['company_name']} data: {e}")


def main():
    """
    Example usage of the risk data processor.
    """
    # Process multiple datasets
    process_multiple_datasets()


if __name__ == "__main__":
    main()
