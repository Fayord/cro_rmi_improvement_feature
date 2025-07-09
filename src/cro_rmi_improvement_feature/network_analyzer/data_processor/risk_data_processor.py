"""
Risk Data Processor

A modular script for processing risk assessment data from Excel files and generating
structured risk data for analysis and visualization.

This module provides functions to:
1. Load and preprocess risk assessment data
2. Calculate risk scores and levels
3. Process and clean data structures
4. Merge with risk catalog data
5. Export processed data to JSON format

Author: Generated from Jupyter notebook
Date: 2025
"""

import os
import json
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional
import warnings

# Suppress pandas warnings about deprecated methods
warnings.filterwarnings("ignore", category=FutureWarning)


class RiskDataProcessor:
    """
    A class to process risk assessment data from Excel files.

    This class encapsulates all the functionality from the original Jupyter notebook
    into modular, reusable methods.
    """

    def __init__(self, selected_risk_categories: Optional[List[str]] = None):
        """
        Initialize the RiskDataProcessor.

        Args:
            selected_risk_categories: List of risk categories to include in processing.
                                    If None, uses default categories.
        """
        self.selected_risk_categories = selected_risk_categories or [
            "Operational Risk",
            "Strategic Risk",
            "Credit Risk",
            "Market Risk",
            "Liquidity Risk",
        ]

        # Column mapping for renaming
        self.column_mapping = {
            "Risk Category": "risk_cat",
            "Risk Item": "risk",
            "Risk Description": "risk_desc",
            "Root Cause": "rootcause",
            "Process": "process",
            "Control": "control_name",
            "Control Description": "control_des",
            "EC-Root Cause": "control_rootcause",
            "EC-Process": "control_process",
        }

        # Final output columns
        self.output_columns = [
            "company",
            "risk_cat",
            "risk",
            "risk_desc",
            "rootcause",
            "process",
            "risk_level",
        ]

    def load_assessment_data(
        self, file_path: str, company_name: str = "PCG"
    ) -> pd.DataFrame:
        """
        Load risk assessment data from Excel file.

        Args:
            file_path: Path to the Excel file containing risk assessment data
            company_name: Name of the company (default: "PCG")

        Returns:
            DataFrame with loaded and preprocessed data
        """
        print(f"Loading assessment data from: {file_path}")

        # Load the Excel file
        df = pd.read_excel(file_path)

        # Add company and empty description columns
        df["company"] = company_name
        df["process_desc"] = ""
        df["rootcause_desc"] = ""

        print(f"Loaded data shape: {df.shape}")
        return df

    def calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores and levels for the dataset.

        Args:
            df: Input DataFrame with likelihood and impact columns

        Returns:
            DataFrame with calculated risk scores and levels
        """
        print("Calculating risk scores...")

        # Calculate risk score
        df["risk_score"] = df["likelihood_combined"] * df["impact_combined"]

        # Calculate risk level
        df["risk_level"] = df.apply(self._determine_risk_level, type="combined", axis=1)

        print(f"Risk level distribution:\n{df['risk_level'].value_counts()}")
        return df

    def _determine_risk_level(self, row: pd.Series, type: str = "combined") -> int:
        """
        Determine risk level based on risk score and impact.

        Args:
            row: DataFrame row containing risk data
            type: Type of risk calculation ("fin", "nonfin", or "combined")

        Returns:
            Risk level (0-4)
        """
        if type == "fin":
            riskscore = "risk_score_fin"
            riskimpact = "impact_fin_combined"
        elif type == "nonfin":
            riskscore = "risk_score_nonfin"
            riskimpact = "impact_nonfin_combined"
        else:
            riskscore = "risk_score"
            riskimpact = "impact_combined"

        score = row[riskscore]
        impact = row[riskimpact]

        if score >= 20:
            return 4
        elif 10 <= score <= 16:
            return 3
        elif 4 <= score <= 9:
            return 2
        elif (0 < score < 4) or (score == 4 and impact == 2):
            return 1
        elif score == 0:
            return 0
        else:
            return 0  # Default case

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to standardized names.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with renamed columns
        """
        print("Renaming columns...")
        df = df.rename(columns=self.column_mapping)
        return df

    def aggregate_risk_data(
        self, df: pd.DataFrame, method: str = "RMI"
    ) -> pd.DataFrame:
        """
        Aggregate risk data by company and risk name.

        Args:
            df: Input DataFrame
            method: Aggregation method ("MAX" or "RMI")

        Returns:
            Aggregated DataFrame
        """
        print(f"Aggregating risk data using method: {method}")

        if method == "MAX":
            return self._aggregate_max_method(df)
        elif method == "RMI":
            return self._aggregate_rmi_method(df)
        else:
            raise ValueError(f"Invalid aggregation method: {method}")

    def _aggregate_max_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate using MAX method."""
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

        # Recalculate risk score
        df2["risk_score"] = df2["impact_combined"] * df2["likelihood_combined"]
        df2["risk_level"] = df2.apply(
            self._determine_risk_level, type="combined", axis=1
        )

        return df2

    def _aggregate_rmi_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate using RMI method."""
        df2 = df.copy()
        df2["risk_score"] = df2["impact_combined"] * df2["likelihood_combined"]

        # Sort by risk score and impact
        df2_sorted = df2.sort_values(
            by=["company", "risk", "risk_score", "impact_combined"],
            ascending=[True, True, False, False],
        )

        # Aggregate
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

        # Recalculate likelihood
        df2["likelihood_combined"] = df2["risk_score"] / df2["impact_combined"]
        df2["risk_level"] = df2.apply(
            self._determine_risk_level, type="combined", axis=1
        )

        return df2

    def filter_companies(self, df: pd.DataFrame, companies: List[str]) -> pd.DataFrame:
        """
        Filter data by company names.

        Args:
            df: Input DataFrame
            companies: List of company names to include

        Returns:
            Filtered DataFrame
        """
        print(f"Filtering companies: {companies}")
        filtered_df = df[df["company"].isin(companies)]
        print(f"Filtered data shape: {filtered_df.shape}")
        return filtered_df

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and reorder columns for final output.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected columns
        """
        print("Selecting output columns...")
        return df[self.output_columns]

    def _list_string_to_list(self, list_string: str) -> Tuple[bool, Union[List, str]]:
        """
        Convert string representation of list to actual list.

        Args:
            list_string: String that might represent a list

        Returns:
            Tuple of (is_list, result)
        """
        is_list = False

        # Check for unbalanced quotes
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
            return is_list, list_cell
        except:
            is_list = False
            return is_list, list_string

    def _cell_to_list(self, cell) -> Union[List, str]:
        """
        Convert cell content to list format.

        Args:
            cell: Cell content (could be set, list, or string)

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
                    is_list, new_list_string_candidate = self._list_string_to_list(
                        list_string_candidate
                    )

                    if not is_list:
                        new_cell.append(new_list_string_candidate)
                        continue

                    for i in new_list_string_candidate:
                        if i in empty_element_list:
                            continue

                        is_list, i = self._list_string_to_list(i)
                        if not is_list:
                            new_cell.append(i)
                            continue
                        new_cell.extend(i)

                except:
                    new_cell.append(list_string_candidate)

            return new_cell

        return cell

    def process_data_structures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean data structures in the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        print("Processing data structures...")

        # Convert sets to lists
        df = df.map(self._cell_to_list)

        # Convert lists to strings
        df = df.map(lambda x: ",".join(x) if isinstance(x, list) else x)

        return df

    def merge_rootcause_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge rootcause and process columns with their descriptions.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with merged columns
        """
        print("Merging rootcause and process columns...")

        # Merge rootcause columns
        df["new_rootcause"] = df.apply(self._merge_rootcause, axis=1)

        # Merge process columns
        df["new_process"] = df.apply(self._merge_process, axis=1)

        # Remove old columns and rename new ones
        df.drop(
            ["rootcause", "rootcause_desc", "process", "process_desc"],
            axis=1,
            inplace=True,
        )
        df.rename(
            columns={"new_rootcause": "rootcause", "new_process": "process"},
            inplace=True,
        )

        return df

    def _merge_rootcause(self, row: pd.Series) -> str:
        """Merge rootcause and rootcause_desc columns."""
        rootcause = row["rootcause"]
        rootcause_desc = row["rootcause_desc"]

        if rootcause == "" and rootcause_desc == "":
            return ""
        elif rootcause == "":
            return "rootcause_desc :" + rootcause_desc
        elif rootcause_desc == "":
            return "rootcause :" + rootcause
        else:
            return (
                "rootcause :" + rootcause + "\n" + "rootcause_desc :" + rootcause_desc
            )

    def _merge_process(self, row: pd.Series) -> str:
        """Merge process and process_desc columns."""
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

    def load_risk_catalog(self, catalog_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load risk catalog data and create mapping dictionary.

        Args:
            catalog_path: Path to the risk catalog Excel file

        Returns:
            Tuple of (risk_catalog_df, risk_to_desc_dict)
        """
        print(f"Loading risk catalog from: {catalog_path}")

        # Load risk catalog data
        risk_df = pd.read_excel(catalog_path, sheet_name="Risks")
        risk_cause_mapping_df = pd.read_excel(
            catalog_path, sheet_name="Risk_Cause_mapping"
        )

        # Fix category name
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

        # Create mapping dictionary
        selected_cols = ["Risk-EN", "Description-EN"]
        risk_and_risk_desc_df = risk_df[selected_cols]
        risk_and_risk_desc_df = risk_and_risk_desc_df.map(lambda x: x.strip())

        risk_to_desc_dict = risk_and_risk_desc_df.set_index("Risk-EN")[
            "Description-EN"
        ].to_dict()

        print(f"Loaded {len(risk_to_desc_dict)} risk mappings")
        return risk_df, risk_to_desc_dict

    def add_catalog_risks(
        self, df: pd.DataFrame, risk_catalog_df: pd.DataFrame, date_stamp: str
    ) -> pd.DataFrame:
        """
        Add risk catalog data to the processed dataset.

        Args:
            df: Processed risk data DataFrame
            risk_catalog_df: Risk catalog DataFrame
            date_stamp: Date stamp for catalog entries

        Returns:
            DataFrame with catalog risks added
        """
        print("Adding risk catalog data...")

        new_rows = []
        for index, row in risk_catalog_df.iterrows():
            risk_cat = row["Risk-category"]
            if risk_cat not in self.selected_risk_categories:
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

        # Concatenate with original data
        result_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"Added {len(new_rows)} catalog risks. Final shape: {result_df.shape}")

        return result_df

    def extract_date_stamp(self, file_path: str) -> str:
        """
        Extract date stamp from file name.

        Args:
            file_path: Path to the file

        Returns:
            Date stamp string
        """
        file_name = os.path.basename(file_path)
        date_stamp = file_name.split("_")[0]
        return date_stamp

    def save_to_json(self, data: List[Dict], output_path: str) -> None:
        """
        Save processed data to JSON file.

        Args:
            data: List of dictionaries containing risk data
            output_path: Path to save the JSON file
        """
        print(f"Saving data to: {output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Successfully saved {len(data)} records to JSON")

    def process_complete_workflow(
        self,
        assessment_file: str,
        catalog_file: str,
        output_file: str,
        companies: Optional[List[str]] = None,
        aggregation_method: str = "RMI",
    ) -> List[Dict]:
        """
        Execute the complete risk data processing workflow.

        Args:
            assessment_file: Path to the risk assessment Excel file
            catalog_file: Path to the risk catalog Excel file
            output_file: Path to save the output JSON file
            companies: List of companies to include (if None, includes all)
            aggregation_method: Method for aggregating risk data ("MAX" or "RMI")

        Returns:
            List of dictionaries containing processed risk data
        """
        print("Starting complete risk data processing workflow...")

        # Step 1: Load assessment data
        df = self.load_assessment_data(assessment_file)

        # Step 2: Calculate risk scores
        df = self.calculate_risk_scores(df)

        # Step 3: Rename columns
        df = self.rename_columns(df)

        # Step 4: Aggregate risk data
        df = self.aggregate_risk_data(df, method=aggregation_method)

        # Step 5: Filter companies (if specified)
        if companies:
            df = self.filter_companies(df, companies)

        # Step 6: Select columns
        df = self.select_columns(df)

        # Step 7: Process data structures
        df = self.process_data_structures(df)

        # Step 8: Merge rootcause and process columns
        df = self.merge_rootcause_and_process(df)

        # Step 9: Load risk catalog
        risk_catalog_df, risk_to_desc_dict = self.load_risk_catalog(catalog_file)

        # Step 10: Add catalog risks
        date_stamp = self.extract_date_stamp(assessment_file)
        df = self.add_catalog_risks(df, risk_catalog_df, date_stamp)

        # Step 11: Convert to list of dictionaries
        result_list = df.to_dict(orient="records")

        # Step 12: Save to JSON
        self.save_to_json(result_list, output_file)

        print("Workflow completed successfully!")
        return result_list


def main():
    """
    Example usage of the RiskDataProcessor class.
    """
    # Initialize processor
    processor = RiskDataProcessor()

    # Define file paths
    dir_path = os.path.dirname(os.path.abspath(__file__))
    assessment_file = os.path.join(
        dir_path, "../data/", "250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx"
    )
    catalog_file = os.path.join(dir_path, "../data/", "RMI-V2-Translate_20250508.xlsx")
    output_file = os.path.join(dir_path, "../result/", "250528-company_risk_data.json")

    # Process the data
    try:
        result = processor.process_complete_workflow(
            assessment_file=assessment_file,
            catalog_file=catalog_file,
            output_file=output_file,
            companies=["PCG"],
            aggregation_method="RMI",
        )

        print(f"Processing completed. Generated {len(result)} risk records.")

    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
