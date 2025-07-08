import os
import pandas as pd
from granularity_classifier import classify_granularity, RISK_CATEGORIES
from collections import Counter
from tqdm import tqdm

if __name__ == "__main__":
    DRY_RUN = True
    DRY_RUN = False

    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, "../data", "seed_excel_13May25.xlsx")
    df = pd.read_excel(data_path)
    print(df.info())
    risk_name_en_list = df["Risk-EN"].tolist()
    risk_desc_en_list = df["Description-EN"].tolist()
    print(f"risk_name_en_list: {len(risk_name_en_list)}")

    result = classify_granularity(
        text=risk_name_en_list[0],
        description=risk_desc_en_list[0],
        context="risk",
        reference_categories=RISK_CATEGORIES,
    )

    print(result)
    if DRY_RUN:
        exit()
    results = []

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
    level_counter = Counter([result["level"] for result in results])
    print(level_counter)
