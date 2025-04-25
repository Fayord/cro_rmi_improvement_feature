import os
import pandas as pd
import json

import os
from typing import List, Literal
import numpy as np

# from nra_utils import (
#     get_embedding_list,
#     scramble_sentences,
#     load_pickle,
#     save_pickle,
#     rank_similarity_embedding_list,
#     create_dynamic_model_for_find_duplicate,
# )

from tqdm import tqdm

import pickle


def get_risk_desc_th_list(risk_data_df_path: str) -> list[str]:
    risk_data_df_dict = pd.read_excel(risk_data_df_path, sheet_name=None)
    risk_data_df = risk_data_df_dict.get("Risks")
    if risk_data_df is None:
        raise ValueError(
            f"Sheet name 'Risks' not found, sheet name list : {risk_data_df_dict.keys()}"
        )
    risk_desc_th_list = risk_data_df["Description-TH"].tolist()
    return risk_desc_th_list


def get_risk_name_and_desc_th_list(risk_data_df_path: str) -> list[str]:
    risk_data_df_dict = pd.read_excel(risk_data_df_path, sheet_name=None)
    risk_data_df = risk_data_df_dict.get("Risks")
    if risk_data_df is None:
        raise ValueError(
            f"Sheet name 'Risks' not found, sheet name list : {risk_data_df_dict.keys()}"
        )
    risk_th_list = risk_data_df["Risk-TH"].tolist()

    risk_desc_th_list = risk_data_df["Description-TH"].tolist()
    risk_name_and_desc_th_list = [
        f"{risk_name_th}\n{risk_desc_th}"
        for risk_name_th, risk_desc_th in zip(risk_th_list, risk_desc_th_list)
    ]
    return risk_name_and_desc_th_list


def get_risk_3_type_to_risk_cat_dict(risk_data_df_path: str) -> dict:
    #
    risk_data_df_dict = pd.read_excel(risk_data_df_path, sheet_name=None)
    risk_data_df = risk_data_df_dict.get("Risks")
    if risk_data_df is None:
        raise ValueError(
            f"Sheet name 'Risks' not found, sheet name list : {risk_data_df_dict.keys()}"
        )

    risk_th_list = risk_data_df["Risk-TH"].tolist()
    risk_desc_th_list = risk_data_df["Description-TH"].tolist()
    risk_name_and_desc_th_list = [
        f"{risk_name_th}\n{risk_desc_th}"
        for risk_name_th, risk_desc_th in zip(risk_th_list, risk_desc_th_list)
    ]
    risk_cat_th_list = risk_data_df["Risk category"].tolist()
    risk_name_to_risk_cat_dict = dict(zip(risk_th_list, risk_cat_th_list))
    risk_desc_to_risk_cat_dict = dict(zip(risk_desc_th_list, risk_cat_th_list))
    risk_name_and_desc_risk_cat_dict = dict(
        zip(risk_name_and_desc_th_list, risk_cat_th_list)
    )
    # merge 3 dict
    risk_3_type_to_risk_cat_dict = {
        **risk_name_to_risk_cat_dict,
        **risk_desc_to_risk_cat_dict,
        **risk_name_and_desc_risk_cat_dict,
    }
    return risk_3_type_to_risk_cat_dict


def get_risk_th_list(risk_data_df_path: str) -> list[str]:
    risk_data_df_dict = pd.read_excel(risk_data_df_path, sheet_name=None)
    risk_data_df = risk_data_df_dict.get("Risks")
    if risk_data_df is None:
        raise ValueError(
            f"Sheet name 'Risks' not found, sheet name list : {risk_data_df_dict.keys()}"
        )
    risk_th_list = risk_data_df["Risk-TH"].tolist()
    return risk_th_list


def get_risk_th_to_risk_desc_th_dict(risk_data_df_path: str) -> dict[str, str]:
    risk_data_df_dict = pd.read_excel(risk_data_df_path, sheet_name=None)
    risk_data_df = risk_data_df_dict.get("Risks")
    if risk_data_df is None:
        raise ValueError(
            f"Sheet name 'Risks' not found, sheet name list : {risk_data_df_dict.keys()}"
        )
    risk_data_df = risk_data_df_dict.get("Risks")
    if risk_data_df is None:
        raise ValueError(
            f"Sheet name 'Risks' not found, sheet name list : {risk_data_df_dict.keys()}"
        )
    risk_th_list = risk_data_df["Risk-TH"].tolist()
    # check is risk_th_list is unique so key in dict will not replaced
    assert len(risk_th_list) == len(set(risk_th_list)), "risk_th_list is not unique"

    risk_desc_th_list = risk_data_df["Description-TH"].tolist()
    risk_th_to_risk_desc_th_dict = dict(zip(risk_th_list, risk_desc_th_list))
    return risk_th_to_risk_desc_th_dict


def get_risk_th_to_risk_name_and_desc_th_dict(risk_data_df_path: str) -> dict[str, str]:
    risk_data_df_dict = pd.read_excel(risk_data_df_path, sheet_name=None)
    risk_data_df = risk_data_df_dict.get("Risks")
    if risk_data_df is None:
        raise ValueError(
            f"Sheet name 'Risks' not found, sheet name list : {risk_data_df_dict.keys()}"
        )
    risk_data_df = risk_data_df_dict.get("Risks")
    if risk_data_df is None:
        raise ValueError(
            f"Sheet name 'Risks' not found, sheet name list : {risk_data_df_dict.keys()}"
        )
    risk_th_list = risk_data_df["Risk-TH"].tolist()
    # check is risk_th_list is unique so key in dict will not replaced
    assert len(risk_th_list) == len(set(risk_th_list)), "risk_th_list is not unique"
    risk_desc_th_list = risk_data_df["Description-TH"].tolist()
    risk_name_th_list = risk_data_df["Risk-TH"].tolist()
    # risk_name_and_desc_th_list is string and string
    risk_name_and_desc_th_list = [
        f"{risk_name_th}\n{risk_desc_th}"
        for risk_name_th, risk_desc_th in zip(risk_name_th_list, risk_desc_th_list)
    ]
    risk_th_to_risk_name_and_desc_th_dict = dict(
        zip(risk_th_list, risk_name_and_desc_th_list)
    )
    return risk_th_to_risk_name_and_desc_th_dict


def preprocess_customer_data_df(customer_data_df: pd.DataFrame) -> pd.DataFrame:
    preprocessed_customer_data_df = customer_data_df.copy()
    # remove column name "(All)"
    preprocessed_customer_data_df = preprocessed_customer_data_df.drop("(All)", axis=1)
    print("before", len(preprocessed_customer_data_df))
    # remove nan
    preprocessed_customer_data_df = preprocessed_customer_data_df.dropna()
    print("after", len(preprocessed_customer_data_df))
    # remove list
    remove_list = ["Row Labels", "Grand Total"]
    preprocessed_customer_data_df = preprocessed_customer_data_df[
        ~(preprocessed_customer_data_df["type"].isin(remove_list))
    ]

    # NOTE: remove (blank) part manually, (blank) is E&C in future version, for now remove it
    remove_blank_part_string = """(blank)
    ไม่ปฏิบัติตามหลักจริยธรรมทางธุรกิจ กับผู้มีส่วนได้ส่วนเสีย
    กฎหมายที่เกี่ยวข้องกับสินค้าข้าวมีการเปลี่ยนแปลง
    การเปลี่ยนแปลงกฎหมาย
    การเปลี่ยนแปลงกฏระเบียบทางการค้า
    การเลิกจ้างไม่สอดคล้องกับพรบ.คุ้มครองแรงงานและหลักจรรยาบรรณธุรกิจ เครือเจริญโภคภัณฑ์
    การผิดสัญญาของคู่ค้า
    การลักทรัพย์ผลผลิต
    ข้อมูลส่วนบุคคลของคู่ค้ารั่วไหล
    ความเสี่ยงจากการที่มีหลายส่วนงานสามารถเข้าถึงข้อมูล CCTV
    ความเสี่ยงจากการลักลอบไม่ปฏิบัติตามกฎระเบียบในการนำของเข้า-ออกจากสถาบันฯ
    สินค้าปลอมแปลง เลียนแบบ
    """
    remove_blank_part_list = [
        i.strip() for i in remove_blank_part_string.split("\n") if i != ""
    ]

    preprocessed_customer_data_df = preprocessed_customer_data_df[
        ~preprocessed_customer_data_df["type"].isin(remove_blank_part_list)
    ]

    return preprocessed_customer_data_df


def get_customer_risk_name_to_risk_desc_dict(
    customer_data_df_path: str,
) -> dict:
    # customer_risk_name_list = get_customer_risk_name_list(customer_data_df_path)
    customer_data_df_dict = pd.read_excel(customer_data_df_path, sheet_name=None)
    value_sheet_customer_data_df = customer_data_df_dict.get("VALUE")
    value_sheet_customer_data_df = value_sheet_customer_data_df[["risk", "risk_desc"]]
    # drop nan
    value_sheet_customer_data_df = value_sheet_customer_data_df.dropna()
    customer_risk_name_list = value_sheet_customer_data_df["risk"].tolist()
    customer_risk_desc_list = value_sheet_customer_data_df["risk_desc"].tolist()

    customer_risk_name_to_risk_desc_dict = dict(
        zip(customer_risk_name_list, customer_risk_desc_list)
    )
    return customer_risk_name_to_risk_desc_dict


def get_customer_risk_3_type_to_risk_cat_dict(
    customer_data_df_path: str,
):
    customer_data_df_dict = pd.read_excel(customer_data_df_path, sheet_name=None)
    value_sheet_customer_data_df = customer_data_df_dict.get("VALUE")
    value_sheet_customer_data_df = value_sheet_customer_data_df[
        ["risk", "risk_cat", "risk_desc"]
    ]
    # drop nan
    value_sheet_customer_data_df = value_sheet_customer_data_df.dropna()
    customer_risk_name_list = value_sheet_customer_data_df["risk"].tolist()
    customer_risk_cat_list = value_sheet_customer_data_df["risk_cat"].tolist()
    customer_risk_desc_list = value_sheet_customer_data_df["risk_desc"].tolist()
    customer_risk_name_and_desc_list = [
        f"{risk_name_th}\n{risk_desc_th}"
        for risk_name_th, risk_desc_th in zip(
            customer_risk_name_list, customer_risk_desc_list
        )
    ]
    customer_risk_3_type_to_risk_desc_dict = {
        **dict(zip(customer_risk_name_list, customer_risk_cat_list)),
        **dict(zip(customer_risk_desc_list, customer_risk_cat_list)),
        **dict(zip(customer_risk_name_and_desc_list, customer_risk_cat_list)),
    }
    return customer_risk_3_type_to_risk_desc_dict


def get_customer_data_pre_mapping_list(customer_data_df_path: str) -> list[str]:
    customer_data_df_dict = pd.read_excel(customer_data_df_path, sheet_name=None)
    value_sheet_customer_data_df = customer_data_df_dict.get("VALUE")
    # print(value_sheet_customer_data_df.head())
    customer_data_df = customer_data_df_dict.get("pivot")
    # print(customer_data_df.columns)
    # print()
    if customer_data_df is None:
        raise ValueError(
            f"Sheet name 'pivot' not found, sheet name list : {customer_data_df_dict.keys()}"
        )
    preprocessed_customer_data_df = preprocess_customer_data_df(customer_data_df)

    customer_data_pre_mapping_list = preprocessed_customer_data_df["type"].tolist()
    target_list = value_sheet_customer_data_df["risk"].tolist()
    count = 0
    customer_data_description_pre_mapping_list = []
    for customer_data_pre_mapping in customer_data_pre_mapping_list:
        # check is customer_data_pre_mapping in value_sheet_customer_data_df column "risk"
        is_data_exist = customer_data_pre_mapping in target_list

        if is_data_exist is False:
            count += 1
            # print(f"{customer_data_pre_mapping=}")
            customer_data_description_pre_mapping = None
        else:

            customer_data_description_pre_mapping = value_sheet_customer_data_df.loc[
                value_sheet_customer_data_df["risk"] == customer_data_pre_mapping
            ]["risk_desc"].values[0]
        customer_data_description_pre_mapping_list.append(
            customer_data_description_pre_mapping
        )

    # print(f"{count=}")
    # rename
    rename_mapping_dict = {
        "การเปลี่ยนแปลงสภาพภูมิอากาศ": "การเปลี่ยนแปลงนโยบายการเปลี่ยนแปลงด้านสภาพภูมิอากาศ",
        "ความเสี่ยงด้านเครดิตของคู่สัญญา (ลูกค้า)": "ความเสี่ยงด้านเครดิตของลูกหนี้ (ภายนอกเครือฯ)",
    }
    customer_data_pre_mapping_list = [
        rename_mapping_dict.get(i, i) for i in customer_data_pre_mapping_list
    ]

    return customer_data_pre_mapping_list, customer_data_description_pre_mapping_list


def get_customer_data_to_risk_th_list(
    customer_data_pre_mapping_list: list[str], risk_th_list: list[str]
) -> dict[str, str]:

    customer_data_to_risk_th_list = []
    for customer_data_pre_mapping in customer_data_pre_mapping_list:
        if customer_data_pre_mapping in risk_th_list:
            value = customer_data_pre_mapping
            continue
        customer_data_to_risk_th_list.append([customer_data_pre_mapping, value])
    return customer_data_to_risk_th_list


def split_risk_th_list_into_existing_and_new(
    risk_th_list: list[str],
) -> tuple[list[str], list[str]]:
    new_risk_th_list = []
    exist_risk_th_list = []
    for risk_th in risk_th_list:

        if ("ทำงาน" in risk_th) or ("พนักงาน" in risk_th):
            new_risk_th_list.append(risk_th)
        else:
            exist_risk_th_list.append(risk_th)
    return exist_risk_th_list, new_risk_th_list


def main():
    IS_FORCED = False
    dir_path = os.path.dirname(os.path.realpath(__file__))
    risk_data_df_path = os.path.join(dir_path, "data/20240811_ALL_risks.xlsx")
    all_risk_th_list = get_risk_th_list(risk_data_df_path)
    print(f"{len(all_risk_th_list)=}")
    customer_data_df_path = os.path.join(
        dir_path, "data/customquestion_mapping_done_update risk detail.xlsx"
    )
    # create a interest df
    customer_data_pre_mapping_list, customer_data_description_pre_mapping_list = (
        get_customer_data_pre_mapping_list(customer_data_df_path)
    )

    # save customer_data_pre_mapping_list for debugging purposes
    debug_file_path = os.path.join(dir_path, "data/debug.json")
    with open(debug_file_path, "w") as f:
        json.dump(customer_data_pre_mapping_list, f, ensure_ascii=False)

    customer_data_to_risk_th_list = get_customer_data_to_risk_th_list(
        customer_data_pre_mapping_list, all_risk_th_list
    )

    customer_data_list = sorted(
        list(set([i[0] for i in customer_data_to_risk_th_list]))
    )
    risk_th_list = sorted(list(set([i[1] for i in customer_data_to_risk_th_list])))

    # print sample of customer_data_to_risk_th_dict 3 keys
    print(
        f"Sample of customer_data_to_risk_th_dict: {customer_data_to_risk_th_list[:3]}"
    )

    augmented_customer_data_dict_save_path = (
        f"{dir_path}/data/augmented_customer_data.json"
    )
    # load_old_augmented_customer_data_dict
    # if augmented_customer_data_dict_save_path exist load it
    if os.path.exists(augmented_customer_data_dict_save_path):

        augmented_customer_data_dict = json.load(
            open(augmented_customer_data_dict_save_path)
        )
    else:
        augmented_customer_data_dict = {}
    for customer_data in tqdm(customer_data_list):
        # alternative_sentences_dict =
        if (
            not IS_FORCED
            and augmented_customer_data_dict.get(customer_data) is not None
        ):
            continue
        alternative_sentences_dict, usage_data = scramble_sentences(customer_data)
        alternative_sentences: list[str] = alternative_sentences_dict[
            "alternative_sentences"
        ]
        augmented_customer_data_dict[customer_data] = alternative_sentences
    # make sure augmented_customer_data_dict have no duplicate data
    check_augment_duplicate = list(augmented_customer_data_dict.keys())
    for customer_data, alternative_sentences in augmented_customer_data_dict.items():
        no_dup_alternative_sentences = []
        for alternative_sentence in alternative_sentences:
            if alternative_sentence not in check_augment_duplicate:
                no_dup_alternative_sentences.append(alternative_sentence)
                check_augment_duplicate.append(alternative_sentence)
        augmented_customer_data_dict[customer_data] = no_dup_alternative_sentences
    # select_risk_th_list_as_exist_and_new_risk_list =
    # save augmented_customer_data_dict to json
    with open(augmented_customer_data_dict_save_path, "w") as f:
        json.dump(augmented_customer_data_dict, f, ensure_ascii=False)

    # exist_risk_th_list, new_risk_th_list = split_risk_th_list_into_existing_and_new(
    #     risk_th_list
    # )
    exist_risk_th_list = risk_th_list.copy()
    all_alternative_sentences = []
    for alternative_sentences in augmented_customer_data_dict.values():
        all_alternative_sentences.extend(alternative_sentences)
    all_text_to_embedding_list = (
        risk_th_list + customer_data_list + all_alternative_sentences
    )
    print(f"{len(all_text_to_embedding_list)=}")
    # assert all element in all_text_to_embedding_list is string
    assert all(isinstance(i, str) for i in all_text_to_embedding_list)
    embedding_dict_path = f"{dir_path}/data/embedding_caches.pkl"
    # load embedding_dict with pickle
    if os.path.exists(embedding_dict_path):
        embedding_dict = load_pickle(embedding_dict_path)
    else:
        embedding_dict = {}
    already_embedding_sentence_list = list(embedding_dict.keys())
    for i in already_embedding_sentence_list:
        print(i)
    # select sentence in all_text_to_embedding_list to embedding that not in already_embedding_sentence_list
    need_to_embedding_sentence_list = [
        str(sentence)
        for sentence in all_text_to_embedding_list
        if sentence not in already_embedding_sentence_list
    ]
    print(len(need_to_embedding_sentence_list))
    if need_to_embedding_sentence_list != []:

        embedding_list = get_embedding_list(need_to_embedding_sentence_list)
        for sentence, embedding in zip(need_to_embedding_sentence_list, embedding_list):
            embedding_dict[sentence] = embedding
        # save embedding_dict with save_pickle
        save_pickle(embedding_dict, embedding_dict_path)

    customer_input_column = []  # sting
    type_column = []  # real, augmented
    risk_th_column = []  # ground truth
    label_map_column = []  # exist, new

    for customer_data_to_risk_th in customer_data_to_risk_th_list:
        customer_data, risk_th = customer_data_to_risk_th
        customer_input_column.append(customer_data)
        type_column.append("real")
        risk_th_column.append(risk_th)

        label_map = "exist"
        label_map_column.append(label_map)
    print(f"{len(customer_input_column)=}")
    ori_process_data_df = pd.DataFrame(
        {
            "customer_input_column": customer_input_column,
            "type_column": type_column,
            "risk_th_column": risk_th_column,
            "label_map_column": label_map_column,
        }
    )
    exist_risk_th_list_embedding_list = [embedding_dict[i] for i in exist_risk_th_list]
    result_dict_list = []
    for top_k in range(1, 5):
        top_k *= 5
        process_data_df = ori_process_data_df.copy()
        for index, row in process_data_df.iterrows():

            # print(row)
            customer_input = row["customer_input_column"]
            # print(customer_input)
            customer_input_embedding = embedding_dict[customer_input]
            similar_indices_list = rank_similarity_embedding_list(
                input_list=[customer_input_embedding],
                reference_list=exist_risk_th_list_embedding_list,
                distance_metric="cosine",
                top_k=top_k,
            )
            similar_indice_list = similar_indices_list[0]
            choice_list = []
            risk_th_column = row["risk_th_column"]
            for similar_indice in similar_indice_list:
                choice = exist_risk_th_list[similar_indice]
                choice_list.append(choice)
                # print(f"{choice=}")
            is_choice_have_target = risk_th_column in choice_list

            choice_list_str = json.dumps(choice_list, ensure_ascii=False)
            similar_indice_list_str = json.dumps(similar_indice_list)
            # print(choice_list_str)
            process_data_df.loc[index, "is_choice_have_target"] = is_choice_have_target
            process_data_df.loc[index, "choices_column"] = choice_list_str
            process_data_df.loc[index, "choices_indices"] = similar_indice_list_str
            DynamicModel = create_dynamic_model_for_find_duplicate(
                customer_input, choice_list
            )
        is_choice_have_target = sum(process_data_df["is_choice_have_target"].tolist())
        total_rows = process_data_df.shape[0]
        acc = round(is_choice_have_target / total_rows * 100, 2)
        result_dict_list.append(
            {
                "top_k": top_k,
                "is_choice_have_target": is_choice_have_target,
                "total_rows": total_rows,
                "accuracy": acc,
            }
        )
    print(result_dict_list)
    # save result_dict to json
    result_dict_save_path = os.path.join(
        dir_path, "result_dict_only_select_choice.json"
    )
    with open(result_dict_save_path, "w") as f:
        json.dump(result_dict_list, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
