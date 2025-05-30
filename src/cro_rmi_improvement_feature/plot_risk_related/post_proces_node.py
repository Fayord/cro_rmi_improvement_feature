# ADD story to top 5 important risk
# ADD direction to all edges
import os
import math
import pickle
import pandas as pd
from collections import Counter

# import function
from plot_network import (
    get_elements_for_company,
    filter_elements_by_weight_and_recalculate_edges,
    find_neighbors,
)
from utils import tell_a_story_risk_data_grouped, classify_edge_relationship

# import variable
from plot_network import (
    real_data_path,
    edge_relationship_path,
    edge_rgb_color_list,
)
from langchain_community.callbacks import get_openai_callback
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


def get_number_edges_to_show(total_nodes):
    # can change later
    return math.ceil(total_nodes * 2)


def save_snapshot(real_data_path):
    # can overried these variable
    print("\n" * 4)
    real_data = pickle.load(open(real_data_path, "rb"))
    checkbox_values = ["risk_desc"]
    # Capture total_edges in the initial call
    companys = sorted({item["company"] for item in real_data})
    companys = [i for i in companys if not i.startswith("risk_catalog")]

    initial_filtered_elements_all_company = []
    # --- Add logic to save initial 10% edges snapshot ---
    snapshot_file_path = f"{real_data_path.replace('.pkl','')}-10percent_edges.pkl"
    if not os.path.exists(snapshot_file_path):
        print(
            f"Snapshot file not found: {snapshot_file_path}. Generating and saving..."
        )
        for company in companys:

            elements, line_weights, total_edges, total_nodes = get_elements_for_company(
                real_data,
                company,
                edge_relationship_path,
                checkbox_values,
            )  # Initial call with empty checklist

            # Calculate initial_num_edges_to_show using the sigmoid function and math.ceil
            initial_num_edges_to_show = get_number_edges_to_show(total_nodes)
            # if total_edges <= 600  initial_num_edges_to_show set to 10% edges
            # if total_edges > 2000 initial_num_edges_to_show set to 5% edges

            # Need to call filter_elements_by_weight_and_recalculate_edges with the initial elements
            # and the weight threshold corresponding to 10% edges.
            # First, find the threshold weight for 10% edges from the initial line_weights
            if total_edges > 0 and initial_num_edges_to_show > 0:
                sorted_weights = sorted(line_weights, reverse=True)
                threshold_weight = sorted_weights[initial_num_edges_to_show - 1]
            elif total_edges == 0 or initial_num_edges_to_show == 0:
                threshold_weight = float(
                    "inf"
                )  # Set a high threshold to show 0 edges if needed
            else:
                threshold_weight = float(
                    "-inf"
                )  # Should not happen if total_edges > 0 and num_edges > 0

            initial_filtered_elements, _ = (
                filter_elements_by_weight_and_recalculate_edges(
                    elements, threshold_weight, edge_rgb_color_list
                )
            )
            initial_filtered_elements_all_company.extend(initial_filtered_elements)
        print("Success all company")
        try:
            with open(snapshot_file_path, "wb") as f:
                pickle.dump(initial_filtered_elements_all_company, f)
            print(
                f"Successfully saved initial 10% edges snapshot to {snapshot_file_path}"
            )
        except Exception as e:
            print(f"Error saving snapshot file: {e}")
    else:
        print(f"Snapshot file already exists: {snapshot_file_path}. Skipping save.")
    # --- End of snapshot logic ---

    return snapshot_file_path


def get_risk_data_from_risk_name(risk_name, risk_data_list):
    for risk in risk_data_list:
        if risk["risk"] == risk_name:
            return risk
    return None


def add_story_to_top_k_node(
    real_data_path,
    snapshot_file_path,
    top_k=5,
):
    raise
    # need to improve this to not use default_company
    raw_risk_data = pickle.load(open(real_data_path, "rb"))
    companys = sorted({item["company"] for item in raw_risk_data})
    companys = [i for i in companys if not i.startswith("risk_catalog")]
    # TODO need to change how we create story
    snapshot_data_list = pickle.load(open(snapshot_file_path, "rb"))

    raw_risk_selected_keys = [
        "risk_cat",
        "risk",
        "risk_desc",
        "rootcause",
        "process",
    ]
    raw_risk_selected_data_list = []
    for i in raw_risk_data:
        raw_risk_selected_data_list.append({k: i[k] for k in raw_risk_selected_keys})

    node_data_list = []
    for node_edge in snapshot_data_list:
        if node_edge["data"].get("source") is not None:
            # it a edge
            continue
        node_data_list.append(node_edge["data"])
    node_data_df = pd.DataFrame(node_data_list)
    node_data_df = node_data_df.sort_values(
        by=["risk_level", "size_level"], ascending=False
    )
    top_k_node_data = node_data_df.head(top_k)
    top_k_node_ids = top_k_node_data["id"].tolist()

    for selected_node_id in top_k_node_ids:
        primary_nodes, _, subgraph_elements = find_neighbors(
            selected_node_id, snapshot_data_list
        )
        input_risk_name = node_data_df[node_data_df["id"] == selected_node_id][
            "label"
        ].values[0]
        related_risk_name_list = node_data_df[node_data_df["id"].isin(primary_nodes)][
            "label"
        ].values.tolist()

        input_risk_data = get_risk_data_from_risk_name(
            input_risk_name, raw_risk_selected_data_list
        )
        related_risk_data_list = [
            get_risk_data_from_risk_name(i, raw_risk_selected_data_list)
            for i in related_risk_name_list
        ]
        story = tell_a_story_risk_data_grouped(input_risk_data, related_risk_data_list)

        for ind in range(len(raw_risk_data)):
            raw_risk_data_item = raw_risk_data[ind]
            if raw_risk_data_item["risk"] == input_risk_name:
                print(f"{input_risk_name=}")
                raw_risk_data[ind]["story"] = story
    print("Done")

    pickle.dump(raw_risk_data, open(real_data_path, "wb"))
    print("save", real_data_path)


def new_finalize_edge_relationship(
    edge_relationship_a_b: dict, edge_relationship_b_a: dict, risk_a: str, risk_b: str
):
    # because I see many case that reason is good enough and reasonable but
    # the classify of edge relationship is not going with the reason
    # so my new idea is to use a->b and b->a 's reason to feed to llm to decide the final relationship
    ...


def finalize_edge_relationship(
    edge_relationship_a_b: dict, edge_relationship_b_a: dict, risk_a: str, risk_b: str
) -> dict:
    def select_final_relationship(
        edge_relationship_a_b: dict, edge_relationship_b_a: dict
    ) -> dict:
        # if it the same
        if (
            edge_relationship_a_b["relationship"]
            == edge_relationship_b_a["relationship"]
        ):
            return edge_relationship_a_b
        # if there is different we will select more relationship
        if edge_relationship_a_b["relationship"] == "no_relationship":
            return edge_relationship_b_a
        if edge_relationship_b_a["relationship"] == "no_relationship":
            return edge_relationship_a_b
        if edge_relationship_a_b["relationship"] == "be_a_cause_to_each_other":
            return edge_relationship_a_b
        if edge_relationship_b_a["relationship"] == "be_a_cause_to_each_other":
            return edge_relationship_b_a
        edge_relation = {
            "relationship": "be_a_cause_to_each_other",
            "reason": edge_relationship_a_b["reason"]
            + "\n\n"
            + edge_relationship_b_a["reason"],
        }
        return edge_relation

    def post_process_relationship(
        final_relationship: dict, risk_a: str, risk_b: str
    ) -> dict:
        if final_relationship["relationship"] in [
            "be_a_cause_to_each_other",
            "no_relationship",
        ]:
            return final_relationship
        if final_relationship["relationship"].find(risk_a) == 0:
            # risk a_caused_by_risk b
            # so it mean risk a is effect and risk b is cause
            final_relationship["relationship"] = "effect_then_cause"
            return final_relationship
        else:
            final_relationship["relationship"] = "cause_then_effect"
            return final_relationship

    final_relationship = select_final_relationship(
        edge_relationship_a_b, edge_relationship_b_a
    )
    return post_process_relationship(final_relationship, risk_a, risk_b)


def add_edge_relationship(
    real_data_path,
    snapshot_file_path,
    edge_relationship_path,
):
    raw_risk_data = pickle.load(open(real_data_path, "rb"))
    companys = sorted({item["company"] for item in raw_risk_data})
    companys = [i for i in companys if not i.startswith("risk_catalog")]

    snapshot_data_list = pickle.load(open(snapshot_file_path, "rb"))

    raw_risk_selected_keys = [
        "risk",
        "risk_desc",
        "rootcause",
    ]
    raw_risk_selected_data_list = []
    for i in raw_risk_data:
        raw_risk_selected_data_list.append({k: i[k] for k in raw_risk_selected_keys})

    node_data_list = []
    edge_data_list = []
    for node_edge in snapshot_data_list:
        if node_edge["data"].get("source") is not None:
            # it a edge
            edge_data_list.append(node_edge["data"])
        else:

            node_data_list.append(node_edge["data"])
    node_data_df = pd.DataFrame(node_data_list)
    relation_counter = Counter()
    relation_result = {}
    new_edge_data_dict = {}
    for edge_data in edge_data_list:
        source_risk_id = edge_data["source"]
        target_risk_id = edge_data["target"]
        source_risk_name = node_data_df[node_data_df["id"] == source_risk_id][
            "label"
        ].values[0]
        target_risk_name = node_data_df[node_data_df["id"] == target_risk_id][
            "label"
        ].values[0]
        print(f"\t{source_risk_name=}")
        print(f"\t{target_risk_name=}")

        risk_a = get_risk_data_from_risk_name(
            source_risk_name, raw_risk_selected_data_list
        )
        risk_b = get_risk_data_from_risk_name(
            target_risk_name, raw_risk_selected_data_list
        )
        edge_relationship_a_b = classify_edge_relationship(risk_a=risk_a, risk_b=risk_b)
        edge_relationship_b_a = classify_edge_relationship(risk_a=risk_b, risk_b=risk_a)
        edge_relationship = finalize_edge_relationship(
            edge_relationship_a_b, edge_relationship_b_a, risk_a["risk"], risk_b["risk"]
        )
        print(f"{type(edge_relationship)=}")
        print(f"\t{edge_relationship=}")
        # key = (
        #     risk_a["risk"],
        #     risk_b["risk"],
        # )
        # sorted_key = tuple(sorted(key))
        # relation_result[sorted_key] = edge_relationship
        # relation_counter[edge_relationship] += 1
        key = tuple((source_risk_name, target_risk_name))
        new_edge_data_dict[key] = {}
        if edge_relationship["relationship"] == "cause_then_effect":
            new_edge_data_dict[key]["direction"] = [1, 0]
        elif edge_relationship["relationship"] == "effect_then_cause":
            new_edge_data_dict[key]["direction"] = [-1, 0]
        elif edge_relationship["relationship"] == "no_relationship":
            new_edge_data_dict[key]["direction"] = [0, 0]
        elif edge_relationship["relationship"] == "be_a_cause_to_each_other":
            new_edge_data_dict[key]["direction"] = [1, -1]
        else:
            raise ValueError(f"Unknown edge_relationship: {edge_relationship}")
        new_edge_data_dict[key]["reason"] = edge_relationship["reason"]
    # save new_edge_data_dict to pickle

    pickle.dump(new_edge_data_dict, open(edge_relationship_path, "wb"))


if __name__ == "__main__":

    snapshot_file_path = save_snapshot(real_data_path)
    print(f"{real_data_path=}")
    # print(f"{default_company=}")
    print(f"{snapshot_file_path=}")
    # add_story_to_top_k_node(
    #     real_data_path,
    #     snapshot_file_path,
    #     top_k=5,
    # )
    # relation_counter=Counter({'no_relationship': 27, 'be_a_cause_to_each_other': 18, 'riskA_cause_riskB': 15})
    #

    with get_openai_callback() as cb:
        add_edge_relationship(
            real_data_path,
            snapshot_file_path,
            edge_relationship_path,
        )

    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
