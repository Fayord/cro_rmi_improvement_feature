import os
import pandas as pd
from granularity_classifier import classify_granularity, RISK_CATEGORIES

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, "../data", "seed_excel_13May25.xlsx")
    df = pd.read_excel(data_path)
    print(df.info())
    risk_name_en_list = df["Risk-EN"].tolist()
    risk_desc_en_list = df["Description-EN"].tolist()
    result = classify_granularity(
        text=risk_name_en_list[0],
        description=risk_desc_en_list[0],
        context="risk",
        reference_categories=RISK_CATEGORIES,
    )
    print(result)
