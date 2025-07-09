import os
import json
import pandas as pd
from typing import Union, List, Dict, Any
import warnings

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
    df["risk_level"] = df.apply(determine_risk_level, type="combined", axis=1)

    # Rename columns
    df.rename(
        columns={
            "Risk Category": "risk_cat",
            "Risk Item": "risk",
            "Risk Description": "risk_desc",
            "Root Cause": "rootcause",
            "Process": "process",
            "Control": "control_name",
            "Control Description": "control_des",
            "EC-Root Cause": "control_rootcause",
            "EC-Process": "control_process",
        },
        inplace=True,
    )

    return df


def preprocess_generic_data(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    """
    Generic preprocessing for other companies.

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

    # Calculate risk score if columns exist
    if "likelihood_combined" in df.columns and "impact_combined" in df.columns:
        df["risk_score"] = df["likelihood_combined"] * df["impact_combined"]
        df["risk_level"] = df.apply(determine_risk_level, type="combined", axis=1)

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
            }
        )
        df2["risk_score"] = df2["impact_combined"] * df2["likelihood_combined"]
        df2["risk_level"] = df2.apply(determine_risk_level, type="combined", axis=1)

    elif score_method == "RMI":
        df2 = df.copy()
        df2["risk_score"] = df2["impact_combined"] * df2["likelihood_combined"]
        df2_sorted = df2.sort_values(
            by=["company", "risk", "risk_score", "impact_combined"],
            ascending=[True, True, False, False],
        )

        df2 = df2_sorted.groupby(["company", "risk"], as_index=False).agg(
            {
                "risk_desc": set,
                "risk_cat": set,
                "impact_combined": "first",
                "risk_score": "max",
                "risk_level": "max",
                "rootcause": set,
                "rootcause_desc": set,
                "process": set,
                "process_desc": set,
            }
        )

        df2["likelihood_combined"] = df2["risk_score"] / df2["impact_combined"]
        df2["risk_level"] = df2.apply(determine_risk_level, type="combined", axis=1)

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
    risk_df = pd.read_excel(catalog_path, sheet_name="Risks")
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
        }
        new_rows.append(data)

    return pd.DataFrame(new_rows), risk_to_desc_dict


def process_risk_data(
    risk_data_path: str,
    catalog_data_path: str,
    company_name: str,
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
    else:
        df = preprocess_generic_data(ori_df, company_name)

    # Get date stamp from filename
    file_name = os.path.basename(risk_data_path)
    date_stamp = file_name.split("_")[0]

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

    # Filter by company
    df3 = df2[df2["company"].isin([company_name])]

    # Select relevant columns
    select_columns = [
        "company",
        "risk_cat",
        "risk",
        "risk_desc",
        "rootcause",
        "rootcause_desc",
        "process",
        "process_desc",
        "risk_level",
    ]
    df4 = df3[select_columns]

    # Convert sets to lists
    df5 = df4.map(cell_to_list)

    # Convert lists to strings
    df6 = df5.map(lambda x: ",".join(x) if isinstance(x, list) else x)

    # Merge rootcause and process columns
    df7 = df6.copy()
    df7["new_rootcause"] = df7.apply(merge_rootcause, axis=1)
    df7["new_process"] = df7.apply(merge_process, axis=1)

    # Clean up columns
    df7.drop(
        ["rootcause", "rootcause_desc", "process", "process_desc"], axis=1, inplace=True
    )
    df7.rename(
        columns={"new_rootcause": "rootcause", "new_process": "process"}, inplace=True
    )

    # Process catalog data
    catalog_df, risk_to_desc_dict = process_catalog_data(
        catalog_data_path, date_stamp, selected_risk_cat
    )

    # Merge with catalog descriptions
    df8 = df7.copy()
    df8["risk"] = df8["risk"].str.strip()
    df8["risk_desc_catalog"] = df8["risk"].map(risk_to_desc_dict)
    df8["risk_desc_catalog"] = df8["risk_desc_catalog"].fillna("")
    df8["risk_desc_catalog"] = df8["risk_desc_catalog"].str.strip()

    # Combine descriptions
    df8["risk_desc"] = df8["risk_desc_catalog"] + " " + df8["risk_desc"]
    df8["risk_desc"] = df8["risk_desc"].str.strip()
    df8.drop("risk_desc_catalog", axis=1, inplace=True)

    # Add catalog data
    df8 = pd.concat([df8, catalog_df], ignore_index=True)

    # Convert to list of dictionaries
    result_list = df8.to_dict(orient="records")

    # Save to JSON if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)

    return result_list


def main():
    """
    Example usage of the risk data processor.
    """
    # Example usage
    risk_data_path = "data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx"
    catalog_data_path = "data/RMI-V2-Translate_20250508.xlsx"
    company_name = "PCG"
    output_path = "result/250528-company_risk_data.json"

    try:
        result = process_risk_data(
            risk_data_path=risk_data_path,
            catalog_data_path=catalog_data_path,
            company_name=company_name,
            output_path=output_path,
        )
        print(f"Successfully processed {len(result)} risk records")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error processing risk data: {e}")


if __name__ == "__main__":
    main()
