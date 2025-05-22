# ADD story to top 5 important risk
# ADD direction to all edges
import os
import math
import pickle
import pandas as pd

# import function
from plot_network import (
    get_elements_for_company,
    filter_elements_by_weight_and_recalculate_edges,
    find_neighbors,
)
from utils import tell_a_story_risk_data_grouped

# import variable
from plot_network import real_data_path, default_company, edge_rgb_color_list


def save_snapshot(real_data_path, default_company):
    # can overried these variable
    print("\n" * 4)
    checkbox_values = ["risk_desc"]
    # Capture total_edges in the initial call
    elements, line_weights, total_edges = get_elements_for_company(
        default_company, checkbox_values
    )  # Initial call with empty checklist

    # --- Add logic to save initial 10% edges snapshot ---
    snapshot_file_path = f"{real_data_path.replace('.pkl','')}-10percent_edges.pkl"

    if not os.path.exists(snapshot_file_path):
        print(
            f"Snapshot file not found: {snapshot_file_path}. Generating and saving..."
        )
        # Get elements filtered to 10% edges
        initial_num_edges_to_show = math.ceil(total_edges * 0.10)
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

        initial_filtered_elements, _ = filter_elements_by_weight_and_recalculate_edges(
            elements, threshold_weight, edge_rgb_color_list
        )

        try:
            with open(snapshot_file_path, "wb") as f:
                pickle.dump(initial_filtered_elements, f)
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


if __name__ == "__main__":

    snapshot_file_path = save_snapshot(real_data_path, default_company)
    print(f"{real_data_path=}")
    print(f"{default_company=}")
    print(f"{snapshot_file_path=}")
    raw_risk_data = pickle.load(open(real_data_path, "rb"))
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
        if node_edge.get("source") is not None:
            # it a edge
            continue
        node_data_list.append(node_edge["data"])
    node_data_df = pd.DataFrame(node_data_list)
    node_data_df = node_data_df.sort_values(
        by=["risk_level", "size_level"], ascending=False
    )
    top_k = 5
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
