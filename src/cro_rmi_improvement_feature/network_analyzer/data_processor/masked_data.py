import os
import pandas as pd


def create_mask_dict_from_excel(
    data_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../data/raw/Stakeholder_Analysis_WJMod.xlsx",
    )
) -> dict:
    df = pd.read_excel(data_path, sheet_name="BusinessGroup_v1")
    mask_dict = {}
    bu_name_list = df["BU name"].tolist()
    # remove nan
    bu_name_list = [x for x in bu_name_list if pd.notna(x)]
    # remove duplicate
    bu_name_list = list(set(bu_name_list))
    # sort bu_name_list by name
    bu_name_list.sort()
    # mask it to company_AA to company_ZZ
    for i, bu_name in enumerate(bu_name_list):
        first_letter = i // 26
        second_letter = i % 26
        mask_dict[bu_name] = (
            f"company_{chr(65+first_letter)}{chr(65+second_letter)}"  # AA to ZZ
        )
    return mask_dict


def mask_data(data: str, mask_dict: dict) -> str:
    for key, value in mask_dict.items():
        # if key is in data, replace it with value and show the key and value
        if key in data:
            print(f"Replacing {key} with {value}")
            data = data.replace(key, value)
    return data


if __name__ == "__main__":

    mask_dict = create_mask_dict_from_excel()
    for key, value in mask_dict.items():
        print(key)
    # print(mask_dict)
