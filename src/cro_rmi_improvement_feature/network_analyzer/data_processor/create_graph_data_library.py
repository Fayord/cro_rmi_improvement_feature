#!/usr/bin/env python3

import time
import json
import itertools
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
from scipy.spatial.distance import pdist, squareform
from data_processor.relation_classifier import classify_relationship
from collections import Counter
from core.models import (
    RiskData,
    RiskDataWithEmbedding,
    RiskOverlayData,
    EdgeData,
    CompanyGraphData,
    GraphDataLibrary,
)


def load_embedded_data(file_path: Path) -> pd.DataFrame:
    """
    Loads data from a pickle file.
    Assumes the data is a dictionary where keys are company names.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def generate_risk_id(company: str, idx: int, date_stamp: str) -> str:
    """Generates a unique node id for a risk data."""
    return f"risk_{company.replace(' ', '_')}_{date_stamp}_{idx}"


def create_nodes_with_embedding(
    data_list: List[Dict[str, Any]], company: str, embedding_key: str
) -> List[RiskDataWithEmbedding]:
    """Creates a list of node dictionaries for Cytoscape."""
    nodes = []
    for idx, data in enumerate(data_list):
        date_stamp = data["date_stamp"]
        node_id = generate_risk_id(company, idx, date_stamp)
        nodes.append(
            RiskDataWithEmbedding(
                data=RiskData(
                    id=node_id,
                    label=data["risk"],
                    risk=data["risk"],
                    risk_cat=data["risk_cat"],
                    risk_level=data["risk_level"],
                    risk_desc=data["risk_desc"],
                    process=data["process"],
                    rootcause=data["rootcause"],
                    risk_desc_summary=data["risk_desc_summary"],
                    process_summary=data["process_summary"],
                    rootcause_summary=data["rootcause_summary"],
                ),
                embedding=data[embedding_key],
            ),
        )

    return nodes


def create_nodes_and_edges(
    nodes: List[RiskDataWithEmbedding],
    company: str,
) -> Tuple[List[RiskDataWithEmbedding], List[Dict[str, Any]]]:
    """Creates nodes and edges for a company."""
    edges = []
    # Store (distance, original_index) to sort distances and then map back to edges
    distance_with_indices = []

    edge_counter = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):

            distance = np.dot(
                nodes[i].embedding,
                nodes[j].embedding,
            )
            # Add a small epsilon to the denominator to prevent division by zero if norm is 0
            norm_i = np.linalg.norm(nodes[i].embedding)
            norm_j = np.linalg.norm(nodes[j].embedding)

            # Handle cases where norm might be zero (e.g., zero vector embedding)
            if norm_i == 0 or norm_j == 0:
                cosine_similarity = 0  # Or define a suitable behavior for zero vectors
            else:
                cosine_similarity = distance / (norm_i * norm_j)

            # The transformation you're testing: (1 - cosine_similarity) / 2
            scaled_distance = (1 - cosine_similarity) / 2

            edges.append(
                {
                    "risk_a_data": nodes[i].data,
                    "risk_b_data": nodes[j].data,
                    "distance": scaled_distance,  # normalize to 0-1
                    "cosine_similarity": cosine_similarity,  # -1 to 1
                }
            )
            distance_with_indices.append((scaled_distance, edge_counter))
            edge_counter += 1

    # Sort distances in descending order (less distance = higher similarity)
    # The rank should be assigned such that the most similar (less distance) has the lowest rank (0)
    sorted_distance_with_indices = sorted(
        distance_with_indices, key=lambda x: x[0], reverse=False
    )

    # Assign similarity rank based on the sorted order
    for rank, (distance, original_index) in enumerate(sorted_distance_with_indices):
        edges[original_index]["similarity_rank"] = rank

    return nodes, edges


def generate_graph_elements_for_company(
    company_data: pd.DataFrame,
    company_name: str,
    classify_model_name: str = "gpt-4.1-mini",
    high_priority_search_space: float = 3.0,  # 3 * number of nodes
    high_priority_atmost_number_edges: int = 3,
) -> Dict[str, CompanyGraphData]:
    """Generates a dictionary of graphs for each valid embedding type for a company."""
    company_graph_datas: Dict[str, CompanyGraphData] = {}
    data_list = company_data.to_dict(orient="records")
    if not data_list:
        return {}
    print(f"data_list[0]: {data_list[0].keys()}")
    embedding_keys = [
        key
        for key in data_list[0].keys()
        if key.startswith("embedding_") and not key.startswith("embedding_text_")
    ]
    print(f"embedding_keys: {embedding_keys}")
    for embedding_key in embedding_keys:
        nodes = create_nodes_with_embedding(data_list, company_name, embedding_key)
        (nodes, edges) = create_nodes_and_edges(nodes, company_name)
        # calculate distance threshold
        number_of_nodes = len(nodes)
        number_of_displayed_edges = 2 * number_of_nodes
        number_of_edges_search_space = high_priority_search_space * number_of_nodes
        processed_edges = []

        # 1. Calculate high_priority for all edges while it is less than high_priority_threshold_percentage
        # high_priority is a list of edges that have high risk level >=3
        # and if one high risk node have less than high_priority_atmost_number_edges edges,
        # then we extend search space to high_priority_search_space * number of nodes
        high_priority_node_counter = Counter()
        for node in nodes:
            if node.data.risk_level >= 3:
                high_priority_node_counter[node.data.id] = 0
        # check is high_priority_node have exceed high_priority_atmost_number_edges then pop it out
        for edge in edges:
            if edge["similarity_rank"] < number_of_displayed_edges:
                if edge["risk_a_data"].id in high_priority_node_counter.keys():
                    high_priority_node_counter[edge["risk_a_data"].id] += 1
                if edge["risk_b_data"].id in high_priority_node_counter.keys():
                    high_priority_node_counter[edge["risk_b_data"].id] += 1
        print()
        print(f"high_priority_node_counter: {high_priority_node_counter}")
        print()
        remain_high_priority_node_counter = Counter()
        for node_id, count in high_priority_node_counter.items():
            # if count is less than high_priority_atmost_number_edges then add it to high_priority_node_list
            if count < high_priority_atmost_number_edges:
                number_of_edges_to_extend = high_priority_atmost_number_edges - count
                remain_high_priority_node_counter[node_id] = number_of_edges_to_extend
        print(remain_high_priority_node_counter)
        print()
        # sort edges by similarity_rank
        edges = sorted(edges, key=lambda x: x["similarity_rank"])
        # set all edge high_priority to False
        for edge in edges:
            edge["high_priority"] = False
        for edge in edges:
            if (
                number_of_displayed_edges
                <= edge["similarity_rank"]
                < number_of_edges_search_space
            ):
                if edge["risk_a_data"].id in remain_high_priority_node_counter.keys():
                    if remain_high_priority_node_counter[edge["risk_a_data"].id] > 0:
                        remain_high_priority_node_counter[edge["risk_a_data"].id] -= 1
                        edge["high_priority"] = True
                if edge["risk_b_data"].id in remain_high_priority_node_counter.keys():
                    if remain_high_priority_node_counter[edge["risk_b_data"].id] > 0:
                        remain_high_priority_node_counter[edge["risk_b_data"].id] -= 1
                        edge["high_priority"] = True
        # count all high priority edges
        print(
            f"Number of high priority edges: {len([edge for edge in edges if edge['high_priority']])}"
        )
        # print edge that have high_priority
        high_priority_edges = [edge for edge in edges if edge["high_priority"]]
        for edge in high_priority_edges:
            print(
                f"high_priority_edges: {edge['risk_a_data'].risk} -> {edge['risk_b_data'].risk} similarity_rank: {edge['similarity_rank']}"
            )

        print(
            f"Number of remain high priority node counter: {remain_high_priority_node_counter}"
        )

        # high_priority_count_per_node = Counter()
        # for edge in edges:
        #     risk_level_a = edge["risk_a_data"].risk_level
        #     risk_level_b = edge["risk_b_data"].risk_level
        #     if risk_level_a >= 3 or risk_level_b >= 3:
        #         edge["high_priority"] = True
        #     else:
        #         edge["high_priority"] = False

        # for edge in edges:
        #     risk_level_a = edge["risk_a_data"].risk_level
        #     risk_level_b = edge["risk_b_data"].risk_level
        #     if risk_level_a >= 3:
        #         edge["high_priority_a_count"] = (
        #             high_priority_count_per_node[edge["risk_a_data"].id] + 1
        #         )
        #     if risk_level_b >= 3:
        #         edge["high_priority_b_count"] = (
        #             high_priority_count_per_node[edge["risk_b_data"].id] + 1
        #         )
        # # print number of high priority edges
        # print(
        #     f"\tNumber of high priority edges: {len([edge for edge in edges if edge['high_priority']])}"
        # )
        # print(
        #     f"\tNumber of low priority edges: {len([edge for edge in edges if not edge['high_priority']])}"
        # )

        # # 2. Separate edges into high-priority and low-priority lists
        # high_priority_edges = sorted(
        #     [edge for edge in edges if edge["high_priority"]],
        #     key=lambda x: x[
        #         "similarity_rank"
        #     ],  # Sort high-priority by similarity_rank too
        # )
        print(f"high_priority_edges: {len(high_priority_edges)}")
        low_priority_edges = sorted(
            [edge for edge in edges if not edge["high_priority"]],
            key=lambda x: x["similarity_rank"],
        )
        print(f"low_priority_edges: {len(low_priority_edges)}")
        # raise
        total_classifications_needed = 2 * number_of_nodes
        classified_count = 0

        # 3. Classify high-priority edges first
        for edge in tqdm(
            high_priority_edges,
            desc=f"Classifying HIGH PRIORITY relationships for {company_name} ({embedding_key})",
        ):
            relationship = classify_relationship(
                edge["risk_a_data"].model_dump(),
                edge["risk_b_data"].model_dump(),
                classify_model_name,
            )
            edge["relationship"] = relationship
            classified_count += 1
            processed_edges.append(edge)  # Add to processed list early to keep track

        # 4. Classify remaining edges from low-priority list until limit is reached
        for edge in tqdm(
            low_priority_edges,
            desc=f"Classifying LOW PRIORITY relationships for {company_name} ({embedding_key})",
        ):
            if classified_count < total_classifications_needed:
                relationship = classify_relationship(
                    edge["risk_a_data"].model_dump(),
                    edge["risk_b_data"].model_dump(),
                    classify_model_name,
                )
                edge["relationship"] = relationship
                classified_count += 1
                processed_edges.append(edge)
            else:
                # If not classified, set relationship to None and add to processed_edges
                edge["relationship"] = {
                    "interdependency_type": None,
                    "direction": None,
                    "rationale": None,
                    "confidence": None,
                }
                processed_edges.append(edge)

        # 5. Process all edges (classified and unclassified) to create EdgeData objects
        final_edge_data_list = []
        for edge in processed_edges:
            relationship = edge.get(
                "relationship",
                {  # Use .get with a default for unclassified edges
                    "interdependency_type": None,
                    "direction": None,
                    "rationale": None,
                    "confidence": None,
                },
            )

            # Determine source and target based on classification only if relationship exists and has a direction
            if relationship["direction"] in ["A → B", "Both"]:
                source = edge["risk_a_data"].id
                target = edge["risk_b_data"].id
            elif relationship["direction"] == "B → A":
                source = edge["risk_b_data"].id
                target = edge["risk_a_data"].id
            else:  # For unclassified or 'None' direction, default to A as source, B as target
                source = edge["risk_a_data"].id
                target = edge["risk_b_data"].id

            final_edge_data_list.append(
                EdgeData(
                    source=source,
                    target=target,
                    interdependency_type=relationship["interdependency_type"],
                    direction=relationship["direction"],
                    rationale=relationship["rationale"],
                    confidence=relationship["confidence"],
                    risk_a_data=edge["risk_a_data"],
                    risk_b_data=edge["risk_b_data"],
                    distance=edge["distance"],
                    similarity_rank=edge["similarity_rank"],
                    cosine_similarity=edge["cosine_similarity"],
                    high_priority=edge["high_priority"],
                )
            )
        print(f"number_of_displayed_edges: {number_of_displayed_edges}")
        company_graph_id = f"{company_name}|{embedding_key}"
        company_graph_datas[company_graph_id] = CompanyGraphData(
            nodes=nodes,
            edges=final_edge_data_list,
            number_of_displayed_edges=number_of_displayed_edges,
        )

    return company_graph_datas


def generate_risk_catalog_top_n_overlay_data(
    risk_catalog_name: str,
    risk_catalog_list: List[RiskDataWithEmbedding],
    company_graph_data: CompanyGraphData,
    top_n: int = 10,
) -> Dict[str, List[RiskDataWithEmbedding]]:
    """Generates overlay data for the risk catalog.
    Dict[str, List[RiskDataWithEmbedding]] where str is the risk name and List[RiskDataWithEmbedding] is the top n relatedrisk data
    """
    time_stamp = risk_catalog_name.split("-")[-1]
    risk_catalog_top_n_overlay_data = {}
    # Ensure all embeddings are numerical lists/arrays
    all_embeddings = np.array([risk_data.embedding for risk_data in risk_catalog_list])
    print(f"{type(company_graph_data)=}")
    nodes: List[RiskDataWithEmbedding] = company_graph_data.nodes
    for risk_data_with_embedding in nodes:
        risk_data: RiskData = risk_data_with_embedding.data
        embedding: List[float] = risk_data_with_embedding.embedding
        embedding = np.array(embedding)
        # print(f"risk_data: {risk_data.keys()}")
        # use this embedding_text_raw_user_data as embbedding data to find the top n related risk data
        # use cosine similarity to find the top n related risk data
        # Use the actual numerical embedding, not the text used to generate it
        # calculate the cosine similarity between the embedding and the risk_catalog_list
        # Reshape the embedding to a 2D array if it's 1D, for dot product compatibility
        cosine_similarities = np.dot(
            embedding.reshape(1, -1), all_embeddings.T
        ).flatten()
        # find the top n related risk data
        top_n_related_risk_data = np.argsort(cosine_similarities)[-top_n:]
        risk_catalog_top_n_overlay_data[risk_data.risk] = []
        for related_risk_data_index in top_n_related_risk_data:
            id = generate_risk_id(
                risk_catalog_name, related_risk_data_index, time_stamp
            )
            label = risk_catalog_list[related_risk_data_index].data.risk
            risk_catalog_top_n_overlay_data[risk_data.risk].append(
                RiskDataWithEmbedding(
                    data=RiskData(
                        id=id,
                        label=label,
                        risk=risk_catalog_list[related_risk_data_index].data.risk,
                        risk_cat=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_cat,
                        risk_level=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_level,
                        risk_desc=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_desc,
                        process=risk_catalog_list[related_risk_data_index].data.process,
                        rootcause=risk_catalog_list[
                            related_risk_data_index
                        ].data.rootcause,
                        risk_desc_summary=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_desc_summary,
                        process_summary=risk_catalog_list[
                            related_risk_data_index
                        ].data.process_summary,
                        rootcause_summary=risk_catalog_list[
                            related_risk_data_index
                        ].data.rootcause_summary,
                    ),
                    embedding=risk_catalog_list[related_risk_data_index].embedding,
                )
            )
    return risk_catalog_top_n_overlay_data


def create_and_save_graphs(data_path: Path, output_dir: Path) -> GraphDataLibrary:
    """
    Loads data, creates graph structures, saves them to files, and returns the result.
    The final structure is {embedding_type: {company_name: {graph}}}.
    """
    df = load_embedded_data(data_path)
    print(f"Loaded {len(df)} rows of data")
    print(df.info())
    company_graph_datas: Dict[str, CompanyGraphData] = {}
    risk_overlay_datas: Dict[str, RiskOverlayData] = {}
    risk_catalog_reference_datas: Dict[str, List[RiskDataWithEmbedding]] = {}
    company_name_list = df["company"].unique()
    risk_catalog_embedding_key = "embedding_summary_user_data"

    for company_name in company_name_list:
        company_data = df[df["company"] == company_name]
        print(f"\tProcessing {company_name} with {len(company_data)} rows")
        if company_name.find("risk_catalog") == -1:
            continue
        timestamp = company_name.split("-")[-1]
        if timestamp not in risk_catalog_reference_datas:
            risk_catalog_reference_datas[timestamp] = []
            risk_catalog_list = company_data.to_dict(orient="records")
            for risk_data in risk_catalog_list:
                id = generate_risk_id(
                    risk_data["company"], risk_data["date_stamp"], risk_data["risk"]
                )
                label = risk_data["risk"]
                risk_catalog_reference_datas[timestamp].append(
                    RiskDataWithEmbedding(
                        data=RiskData(
                            id=id,
                            label=label,
                            **risk_data,
                        ),
                        embedding=risk_data[risk_catalog_embedding_key],
                    )
                )
    for company_name in company_name_list[::-1]:

        if company_name.find("risk_catalog") != -1:
            continue
        company_data = df[df["company"] == company_name]
        company_graph_data_dict: Dict[str, CompanyGraphData] = (
            generate_graph_elements_for_company(company_data, company_name)
        )

        company_graph_datas.update(company_graph_data_dict)
        # create risk overlay data
        for company_graph_id, company_graph_data in company_graph_data_dict.items():
            overlay_key = f"risk_catalog_top_n_overlay|{timestamp}|{company_graph_id}"
            risk_catalog_top_n_overlay_data = generate_risk_catalog_top_n_overlay_data(
                company_name,
                risk_catalog_reference_datas[timestamp],
                company_graph_data,
                top_n=10,
            )
            risk_overlay_datas[overlay_key] = RiskOverlayData(
                overlay_data=risk_catalog_top_n_overlay_data,
            )

    graph_data_library = GraphDataLibrary(
        company_graph_datas=company_graph_datas,
        risk_overlay_datas=risk_overlay_datas,
        risk_catalog_reference_datas=risk_catalog_reference_datas,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # save in one single file save to pickle
    with open(output_dir / "graph_data_library.pkl", "wb") as f:
        pickle.dump(graph_data_library, f)
    print(f"Saved all graphs to {output_dir / 'graph_data_library.pkl'}")
    return graph_data_library


if __name__ == "__main__":
    # Assuming this script is in src/cro_rmi_improvement_feature/network_analyzer/data_processor/
    # And the data is in src/cro_rmi_improvement_feature/network_analyzer/data/
    SCRIPT_DIR = Path(__file__).parent.resolve()
    BASE_DIR = SCRIPT_DIR.parent.parent
    DATA_PATH = (
        BASE_DIR
        / "network_analyzer"
        / "data"
        / "embeddings"
        / "riskview_data_with_embeddings.pkl"
    )
    OUTPUT_DIR = BASE_DIR / "network_analyzer" / "data" / "graph"

    print(f"Loading data from: {DATA_PATH}")
    print(f"Will save graphs to: {OUTPUT_DIR}")

    create_and_save_graphs(DATA_PATH, OUTPUT_DIR)
    print("\nGraph creation process completed successfully.")
