from collections import Counter
from utils import (
    find_proportional_count_boundaries,
    get_level_from_boundaries,
)
import random
import os
import pickle
import numpy as np
from plot_network_defaut_value import (
    EDGE_SIZE_MULTIPLIER,
    edge_rgb_color_list,
    rgb_color_list,
    risk_cat_color_dict,
)
from utils import find_equal_count_boundaries
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


# Initial elements for the default company
def get_elements_for_company(
    data,
    company,
    edge_relationship_path,
    selected_checklist_values,
):
    filtered = [item for item in data if item["company"] == company]
    if filtered:
        # Capture total_edges from the return value
        elements, line_weights, total_edges, total_nodes = (
            generate_network_from_real_data(
                company,
                filtered,
                edge_relationship_path,
                selected_checklist_values,
            )
        )
        return elements, line_weights, total_edges, total_nodes
    else:
        return [], [], 0, 0  # Return 0 total edges if no data


# New function to filter elements by weight and recalculate edge properties
def filter_elements_by_weight_and_recalculate_edges(
    elements, slider_value, edge_rgb_color_list
):
    node_edge_counter = Counter()
    filtered_elements = []
    old_line_weights = []
    nodes_in_filtered = {}  # Store nodes for easy access

    # First pass: Filter edges and collect raw weights of visible edges
    for el in elements:
        if "source" in el.get("data", {}):
            if el["data"]["raw_weight"] >= slider_value:
                # Append edge for now, will update properties in second pass
                filtered_elements.append(el)
                node_edge_counter["edge"] += 1
                old_line_weights.append(el["data"]["raw_weight"])
        else:
            # Keep all nodes and store them
            filtered_elements.append(el)
            nodes_in_filtered[el["data"]["id"]] = el
            node_edge_counter["node"] += 1

    # Recalculate edge boundaries and update edge properties for visible edges
    if old_line_weights:
        edge_boudaries = find_proportional_count_boundaries(
            old_line_weights, [60, 30, 10]
        )
        for el in filtered_elements:
            if "source" in el.get("data", {}):
                old_line_weight = el["data"]["raw_weight"]
                level = get_level_from_boundaries(edge_boudaries, old_line_weight)
                display_weight = level * EDGE_SIZE_MULTIPLIER
                el["data"]["color"] = edge_rgb_color_list[level - 1]
                el["data"][
                    "weight"
                ] = display_weight  # Update edge weight to display weight

    # --- New logic to recalculate node raw_size based on filtered edge display weights ---
    # Initialize node raw sizes based on filtered edges
    node_raw_sizes = {node_id: 0.0 for node_id in nodes_in_filtered.keys()}

    for el in filtered_elements:
        if "source" in el.get("data", {}):
            src_id = el["data"]["source"]
            tgt_id = el["data"]["target"]
            # Use the updated display weight ('weight')
            w = el["data"]["weight"]
            if src_id in node_raw_sizes:
                node_raw_sizes[src_id] += w
            if tgt_id in node_raw_sizes:
                node_raw_sizes[tgt_id] += w

    # Get the list of raw sizes in the same order as nodes were added to filtered_elements
    current_node_raw_sizes = [
        node_raw_sizes[el["data"]["id"]]
        for el in filtered_elements
        if "source" not in el.get("data", {})
    ]
    node_size_counter = Counter()
    # Recalculate node boundaries and update node sizes for visible nodes
    if current_node_raw_sizes:
        # Assuming node_proportion_list and node_size_list are accessible in this scope
        # (They are defined globally in the provided context)
        node_proportion_list = [65, 30, 5]  # Define or ensure access to these
        node_size_list = [1, 50, 120]  # Define or ensure access to these

        node_boudaries = find_proportional_count_boundaries(
            current_node_raw_sizes, node_proportion_list
        )

        node_idx = 0
        for el in filtered_elements:
            if "source" not in el.get("data", {}):
                raw_size = current_node_raw_sizes[node_idx]
                level = get_level_from_boundaries(node_boudaries, raw_size)
                display_size = node_size_list[level - 1]
                node_size_counter[level] += 1
                el["data"]["size"] = display_size  # Update node size to display size
                node_idx += 1
    # --- End of new logic ---
    print(f"{node_size_counter=}")
    return filtered_elements, node_edge_counter


def generate_network_from_real_data(
    company,
    data_list,
    edge_relationships_data_path,
    selected_checklist_values=None,
):
    """
    data_list: list of dicts, each dict contains:
        - "risk": str
        - "embedding_risk": list or np.array
        - "risk_desc": str
        - "embedding_risk_desc": list or np.array
    """
    # try:
    #     dir_path = os.path.dirname(os.path.realpath(__file__))
    # except Exception:
    #     dir_path = os.getcwd()
    if os.path.exists(edge_relationships_data_path):
        edge_relationships_data = pickle.load(open(edge_relationships_data_path, "rb"))
    else:
        edge_relationships_data = {}
    node_size_multiplier = 10
    number_of_scales = 3
    node_proportion_list = [65, 30, 5]
    node_size_counter = Counter()
    assert len(node_proportion_list) == number_of_scales
    # node_size_list = [1, 30, 60]

    node_size_list = [1, 50, 120]
    assert len(node_size_list) == number_of_scales
    nodes = []
    edges = []
    line_weight_list = []
    # print checkbox value in this function
    selected_checkbox_value_list = []
    if selected_checklist_values is None:
        selected_checklist_values = []
    for selected_checklist_value in selected_checklist_values:
        selected_checkbox_value_list.append(selected_checklist_value)

    # Create edges based on embedding distance (Euclidean)
    # cal the distance between each pair of risks first
    stored_distances = {}
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            emb1 = np.array(
                data_list[i][
                    tuple(sorted(["risk", "embedding"] + selected_checkbox_value_list))
                ]
            )
            emb2 = np.array(
                data_list[j][
                    tuple(sorted(["risk", "embedding"] + selected_checkbox_value_list))
                ]
            )
            distance = np.linalg.norm(emb1 - emb2)
            distance = 1 / distance * 100
            line_weight_list.append(distance)
            stored_distances[(i, j)] = distance

    min_dist = min(line_weight_list)
    max_dist = max(line_weight_list)
    avg_dist = np.mean(line_weight_list)
    print(f"{avg_dist=}")

    edge_boudaries = find_equal_count_boundaries(line_weight_list, number_of_scales)
    # then create edges
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            raw_weight = stored_distances[(i, j)]
            # display_weight is seperated into 3 scales based on the min_dist and max_dist
            level = get_level_from_boundaries(edge_boudaries, raw_weight)
            display_weight = level * EDGE_SIZE_MULTIPLIER
            edge_color = edge_rgb_color_list[level - 1]
            edge_relation_i_j = edge_relationships_data.get(
                tuple((data_list[i]["risk"], data_list[j]["risk"], company)), None
            )
            edge_relation_j_i = edge_relationships_data.get(
                tuple((data_list[j]["risk"], data_list[i]["risk"], company)), None
            )
            if edge_relation_i_j is None and edge_relation_j_i is None:
                # no relationship
                edge_relation = [0, 0]
                edge_relation_reason = ""
            elif edge_relation_i_j is not None and edge_relation_j_i is None:
                edge_relation = edge_relation_i_j["direction"]
                edge_relation_reason = edge_relation_i_j["reason"]
                if edge_relation[0] == 1:
                    i, j = i, j
                elif edge_relation[0] == -1:
                    i, j = j, i
            elif edge_relation_i_j is None and edge_relation_j_i is not None:
                edge_relation = edge_relation_j_i["direction"]
                edge_relation_reason = edge_relation_j_i["reason"]
                if edge_relation[0] == 1:
                    i, j = j, i
                elif edge_relation[0] == -1:
                    i, j = i, j

            else:
                print("error")
                print(data_list[i]["risk"])
                print(data_list[j]["risk"])
                print(edge_relation_i_j)
                print(edge_relation_j_i)
                raise ValueError("it should have 1 None")
            # print(f"{edge_relation_i_j=},{edge_relation_j_i=}")
            source_risk_data = {
                "risk": data_list[i]["risk"],
                "risk_desc": data_list[i]["risk_desc"],
                "risk_level": data_list[i]["risk_level"],
                "risk_cat": data_list[i]["risk_cat"],
                "process": data_list[i].get("process", ""),
                "rootcause": data_list[i].get("rootcause", ""),
            }
            target_risk_data = {
                "risk": data_list[j]["risk"],
                "risk_desc": data_list[j]["risk_desc"],
                "risk_level": data_list[j]["risk_level"],
                "risk_cat": data_list[j]["risk_cat"],
                "process": data_list[i].get("process", ""),
                "rootcause": data_list[i].get("rootcause", ""),
            }
            edges.append(
                {
                    "data": {
                        "source": f"risk_{company}_{i}",
                        "target": f"risk_{company}_{j}",
                        "weight": display_weight,
                        "raw_weight": raw_weight,
                        "color": edge_color,
                        "arrow_weight": (
                            "none" if edge_relation[0] == 0 else "triangle"
                        ),
                        "do_not_cal_weight": False,
                        "edge_relation_reason": edge_relation_reason,
                        "source_risk_data": source_risk_data,
                        "target_risk_data": target_risk_data,
                    }
                }
            )
            if edge_relation[1] == -1:
                edges.append(
                    {
                        "data": {
                            "source": f"risk_{company}_{j}",
                            "target": f"risk_{company}_{i}",
                            "weight": display_weight,
                            "raw_weight": raw_weight,
                            "color": edge_color,
                            "arrow_weight": "triangle",
                            "do_not_cal_weight": True,
                            "edge_relation_reason": edge_relation_reason,
                            "source_risk_data": target_risk_data,
                            "target_risk_data": source_risk_data,
                        }
                    }
                )

    # Calculate raw_size for each node (sum of all connected edge weights)
    node_raw_sizes = [0.0 for _ in range(len(data_list))]

    for edge in edges:
        if edge["data"]["do_not_cal_weight"]:
            continue
        src = int(edge["data"]["source"].split("_")[-1])
        tgt = int(edge["data"]["target"].split("_")[-1])
        w = edge["data"]["raw_weight"]
        node_raw_sizes[src] += w
        node_raw_sizes[tgt] += w

    node_boudaries = find_proportional_count_boundaries(
        node_raw_sizes, node_proportion_list
    )
    # Scale node sizes for display (between 20 and 100)
    min_raw = min(node_raw_sizes)
    max_raw = max(node_raw_sizes)

    def scale_size(raw):
        if max_raw == min_raw:
            return 60  # fallback if all are equal
        return 20 + (raw - min_raw) / (max_raw - min_raw) * (100 - 20)

    # Create nodes for each risk
    for idx, data in enumerate(data_list):
        raw_size = node_raw_sizes[idx]
        level = get_level_from_boundaries(node_boudaries, raw_size)
        node_size_counter[level] += 1
        display_size = node_size_list[level - 1]
        risk_cat = data["risk_cat"]
        risk_cat_color = risk_cat_color_dict[risk_cat]
        nodes.append(
            {
                "data": {
                    "id": f"risk_{company}_{idx}",
                    "label": data["risk"],
                    "raw_size": raw_size,
                    "size_level": level,
                    "size": display_size,
                    "color": risk_cat_color,
                    "risk_level": data["risk_level"],
                    "story": data.get("story", ""),
                },
                "position": {
                    "x": random.uniform(100, 700),
                    "y": random.uniform(100, 700),
                },
            }
        )
    print(f"{node_size_counter=}")
    # Return nodes, edges, line_weight_list, and total number of edges
    # save nodes and edges to pickle file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(f"{dir_path}/result/nodes_and_edges_{company}.pkl", "wb") as f:
        pickle.dump([nodes, edges], f)
    return nodes + edges, line_weight_list, len(edges), len(nodes)
