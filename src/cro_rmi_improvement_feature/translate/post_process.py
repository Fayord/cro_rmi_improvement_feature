import pandas as pd
import os
import re

if __name__ == "__main__":
    
    # Path configuration
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "plan_control_name_translated.xlsx"
    output_file_name = "plan_control_name_translated_post_process.xlsx"
    
    output_file_path = os.path.join(dir_path, output_file_name)
    file_path = os.path.join(dir_path, file_name)

    print(f"Reading file: {file_path}")
    
    # Read all sheets
    df_dict = pd.read_excel(file_path, sheet_name=None)
    
    # List to store report of empty cells
    empty_cells_report = []

    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            
            print(f"Processing sheet: {sheet_name}")

            # Check if dataframe has at least 3 columns (Index 0, 1, 2)
            if len(df.columns) > 2:
                # Define column names based on index
                col_source = df.columns[1] # Column 2 (Source)
                col_target = df.columns[2] # Column 3 (Target/Translation)
                
                # 1. Ensure columns are strings
                df[col_source] = df[col_source].astype(str).replace('nan', '')
                df[col_target] = df[col_target].astype(str).replace('nan', '')

                # 2. CLEANING LOGIC (Remove "without any..." phrases from Target)
                cleanup_pattern = r"(without any other text or explanation\.?|without any additional text or explanation\.?)"
                df[col_target] = df[col_target].str.replace(cleanup_pattern, "", regex=True, case=False)

                # 3. ENGLISH CHECK LOGIC
                # Check if Source (Col 2) is purely English (ASCII characters only)
                # ^ starts string, [\x00-\x7F] is the ASCII range, + means one or more, $ ends string
                # This returns True for "Hello World" or "System_ID_123", but False for "สวัสดี"
                is_english_mask = df[col_source].str.contains(r'^[\x00-\x7F]+$', regex=True, na=False)
                
                # If Source is English, overwrite Target with Source
                df.loc[is_english_mask, col_target] = df.loc[is_english_mask, col_source]

                # 4. Final Strip (Whitespace cleanup)
                df[col_target] = df[col_target].str.strip()

                # 5. CHECK FOR EMPTY CELLS (In Target Column)
                # Check for empty strings ""
                empty_mask = (df[col_target] == "")
                
                if empty_mask.any():
                    empty_rows = df.index[empty_mask].tolist()
                    for row_idx in empty_rows:
                        # Log the location (Row + 2 to match Excel row numbers)
                        empty_cells_report.append(f"Sheet: '{sheet_name}' | Row: {row_idx + 2}")

            # Write to the specific sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Processing complete. Saved to: {output_file_path}")
    
    # Print the report
    print("-" * 40)
    if empty_cells_report:
        print(f"Found {len(empty_cells_report)} empty cells in Column 3:")
        for item in empty_cells_report:
            print(item)
    else:
        print("No empty cells found in Column 3 after processing.")
    print("-" * 40)