#!/usr/bin/env python3

import time
import json
import itertools
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from langchain_community.callbacks import get_openai_callback

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
from core.models import (
    RiskDataNoEmbedding,
    RiskDataWithOutEmbedding,
    EdgeDataNoEmbedding,
    CompanyGraphDataNoEmbedding,
    GraphDataLibraryNoEmbedding,
    RiskOverlayDataNoEmbedding,
)
import igraph as ig
import leidenalg
from api.utils import get_networkx_graph_from_company_graph_data


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


def convert_to_no_embedding_risk_data(
    risk_data_with_embedding: RiskDataWithEmbedding,
) -> RiskDataWithOutEmbedding:
    """Converts RiskDataWithEmbedding to RiskDataWithOutEmbedding."""
    # Create an instance of RiskDataNoEmbedding from the data part of RiskDataWithEmbedding
    risk_data_no_embedding = RiskDataNoEmbedding(
        **risk_data_with_embedding.data.model_dump()
    )
    return RiskDataWithOutEmbedding(data=risk_data_no_embedding)


def convert_to_no_embedding_edge_data(edge_data: EdgeData) -> EdgeDataNoEmbedding:
    """Converts EdgeData to EdgeDataNoEmbedding."""
    return EdgeDataNoEmbedding(
        source=edge_data.source,
        target=edge_data.target,
        interdependency_type=edge_data.interdependency_type,
        direction=edge_data.direction,
        rationale=edge_data.rationale,
        confidence=edge_data.confidence,
        risk_a_data=RiskDataNoEmbedding(**edge_data.risk_a_data.model_dump()),
        risk_b_data=RiskDataNoEmbedding(**edge_data.risk_b_data.model_dump()),
        distance=edge_data.distance,
        cosine_similarity=edge_data.cosine_similarity,
        high_priority=edge_data.high_priority,
        similarity_rank=edge_data.similarity_rank,
    )


def convert_to_no_embedding_company_graph_data(
    company_graph_data: CompanyGraphData,
) -> CompanyGraphDataNoEmbedding:
    """Converts CompanyGraphData to CompanyGraphDataNoEmbedding."""
    nodes_no_embedding = [
        convert_to_no_embedding_risk_data(node) for node in company_graph_data.nodes
    ]
    edges_no_embedding = [
        convert_to_no_embedding_edge_data(edge) for edge in company_graph_data.edges
    ]
    return CompanyGraphDataNoEmbedding(
        nodes=nodes_no_embedding,
        edges=edges_no_embedding,
        number_of_displayed_edges=company_graph_data.number_of_displayed_edges,
        risk_catalog_reference_id=company_graph_data.risk_catalog_reference_id,
        overlay_top_n_risks_catalog_id=company_graph_data.overlay_top_n_risks_catalog_id,
        overlay_news_id_list=company_graph_data.overlay_news_id_list,
    )


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
                    risk_id=node_id,
                    risk_cat=data["risk_cat"],
                    risk_level=data["risk_level"],
                    risk_score=data["risk_score"],
                    risk_impact=data["impact_combined"],
                    risk_likelihood=data["likelihood_combined"],
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


def _prioritize_edges(
    nodes: List[RiskDataWithEmbedding],
    edges: List[Dict],
    number_of_displayed_edges: int,
    high_priority_search_space: float,
    high_priority_atmost_number_edges: int,
) -> List[Dict]:
    """
    Prioritizes edges based on risk level and proximity.

    Args:
        nodes: List of RiskNode objects.
        edges: List of dictionaries representing edges.
        number_of_displayed_edges: The number of edges to be displayed initially.
        high_priority_search_space: Multiplier for the search space of high-priority edges.
        high_priority_atmost_number_edges: Maximum number of high-priority edges per node.

    Returns:
        A list of edges with 'high_priority' flag set.
    """
    number_of_nodes = len(nodes)
    number_of_edges_search_space = high_priority_search_space * number_of_nodes

    high_priority_node_counter = Counter()
    for node in nodes:
        if node.data.risk_level >= 3:
            high_priority_node_counter[node.data.id] = 0

    for edge in edges:
        if edge["similarity_rank"] < number_of_displayed_edges:
            if edge["risk_a_data"].id in high_priority_node_counter.keys():
                high_priority_node_counter[edge["risk_a_data"].id] += 1
            if edge["risk_b_data"].id in high_priority_node_counter.keys():
                high_priority_node_counter[edge["risk_b_data"].id] += 1

    remain_high_priority_node_counter = Counter()
    for node_id, count in high_priority_node_counter.items():
        if count < high_priority_atmost_number_edges:
            number_of_edges_to_extend = high_priority_atmost_number_edges - count
            remain_high_priority_node_counter[node_id] = number_of_edges_to_extend

    edges = sorted(edges, key=lambda x: x["similarity_rank"])
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
    return edges


def _classify_edges(
    high_priority_edges: List[Dict],
    low_priority_edges: List[Dict],
    company_name: str,
    embedding_key: str,
    classify_model_name: str,
    relation_process: str,
    total_classifications_needed: int,
) -> Tuple[List[Dict], Counter, List[Dict], int]:
    """
    Classifies relationships for high-priority and low-priority edges.

    Args:
        high_priority_edges: List of high-priority edges.
        low_priority_edges: List of low-priority edges.
        company_name: Name of the company.
        embedding_key: Key for the embedding type.
        classify_model_name: Name of the classification model.
        relation_process: Type of relation processing ("oneway_run" or "twoway_run").
        total_classifications_needed: Total number of classifications to perform.

    Returns:
        Tuple containing processed_edges, direction_list, edge_label_list, and classified_count.
    """
    processed_edges = []
    classified_count = 0
    direction_list = Counter()
    edge_label_list = []

    # Classify high-priority edges first
    for edge in tqdm(
        high_priority_edges,
        desc=f"Classifying HIGH PRIORITY relationships for {company_name} ({embedding_key})",
    ):
        relationship_a_b = classify_relationship(
            edge["risk_a_data"].model_dump(),
            edge["risk_b_data"].model_dump(),
            classify_model_name,
        )
        direction_list[relationship_a_b["direction"]] += 1

        if relation_process == "twoway_run":
            relationship_b_a = classify_relationship(
                edge["risk_b_data"].model_dump(),
                edge["risk_a_data"].model_dump(),
                classify_model_name,
            )
            edge["relationship"] = process_two_way_relationships(
                relationship_a_b, relationship_b_a
            )
        else:
            edge["relationship"] = relationship_a_b
        classified_count += 1
        processed_edges.append(edge)

    edge_label_counter = 0
    total_edge_label_counter = int(len(low_priority_edges) * 0.2)

    # Classify remaining edges from low-priority list until limit is reached
    for edge in tqdm(
        low_priority_edges,
        desc=f"Classifying LOW PRIORITY relationships for {company_name} ({embedding_key})",
    ):
        if classified_count < total_classifications_needed:
            relationship_a_b = classify_relationship(
                edge["risk_a_data"].model_dump(),
                edge["risk_b_data"].model_dump(),
                classify_model_name,
            )
            direction_list[relationship_a_b["direction"]] += 1

            if relation_process == "twoway_run":
                relationship_b_a = classify_relationship(
                    edge["risk_b_data"].model_dump(),
                    edge["risk_a_data"].model_dump(),
                    classify_model_name,
                )
                edge["relationship"] = process_two_way_relationships(
                    relationship_a_b, relationship_b_a
                )
                edge_label_counter += 1
                if edge_label_counter < total_edge_label_counter:
                    count_possible_missing_direction_relation = None
                    if (
                        edge["relationship"]["interdependency_type"] in ["Causal"]
                    ) and (relationship_a_b["interdependency_type"] not in ["Causal"]):
                        count_possible_missing_direction_relation = True
                    if relationship_a_b["interdependency_type"] in [
                        "Causal",
                    ]:
                        count_possible_missing_direction_relation = False
                    risk_a_data_str = ""
                    for key, value in edge["risk_a_data"].model_dump().items():
                        risk_a_data_str += f"{key}: {value}\n"
                    risk_b_data_str = ""
                    for key, value in edge["risk_b_data"].model_dump().items():
                        risk_b_data_str += f"{key}: {value}\n"
                    edge_label_data_a_b = {
                        "direction_risk": "a->b",
                        "analyze_model_name": "gpt-4.1-mini",
                        "source_risk": edge["risk_a_data"].risk,
                        "source_risk_data": risk_a_data_str,
                        "target_risk": edge["risk_b_data"].risk,
                        "target_risk_data": risk_b_data_str,
                        "interdependency_type": relationship_a_b[
                            "interdependency_type"
                        ],
                        "direction": relationship_a_b["direction"],
                        "rationale": relationship_a_b["rationale"],
                        "confidence": relationship_a_b["confidence"],
                        "final_interdependency_type": edge["relationship"][
                            "interdependency_type"
                        ],
                        "final_direction": edge["relationship"]["direction"],
                        "count_possible_missing_direction_relation": (
                            count_possible_missing_direction_relation
                        ),
                    }
                    edge_label_data_b_a = {
                        "direction_risk": "b->a",
                        "analyze_model_name": "gpt-4.1-mini",
                        "source_risk": edge["risk_a_data"].risk,
                        "source_risk_data": risk_a_data_str,
                        "target_risk": edge["risk_b_data"].risk,
                        "target_risk_data": risk_b_data_str,
                        "interdependency_type": relationship_b_a[
                            "interdependency_type"
                        ],
                        "direction": relationship_b_a["direction"],
                        "rationale": relationship_b_a["rationale"],
                        "confidence": relationship_b_a["confidence"],
                        "final_interdependency_type": edge["relationship"][
                            "interdependency_type"
                        ],
                        "final_direction": edge["relationship"]["direction"],
                        "count_possible_missing_direction_relation": (
                            count_possible_missing_direction_relation
                        ),
                    }
                    edge_label_list.append(edge_label_data_a_b)
                    edge_label_list.append(edge_label_data_b_a)
            else:
                edge["relationship"] = relationship_a_b
            classified_count += 1
            processed_edges.append(edge)
        else:
            edge["relationship"] = {
                "interdependency_type": None,
                "direction": None,
                "rationale": None,
                "confidence": None,
            }
            processed_edges.append(edge)

    return processed_edges, direction_list, edge_label_list, classified_count


def _create_final_edge_data(processed_edges: List[Dict]) -> List[EdgeData]:
    """
    Converts processed edges into a list of EdgeData objects.

    Args:
        processed_edges: List of dictionaries representing processed edges.

    Returns:
        A list of EdgeData objects.
    """
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
    return final_edge_data_list


def get_number_displayed_edges(number_of_nodes: int) -> int:
    """Get the number of displayed edges based on the number of nodes."""
    return math.ceil(2.5 * number_of_nodes)


def _process_edges(
    nodes: List[RiskDataWithEmbedding],
    company_name: str,
    embedding_key: str,
    classify_model_name: str,
    relation_process: str,
    high_priority_search_space: float,
    high_priority_atmost_number_edges: int,
) -> Tuple[List[Dict], Counter, List[Dict], int, List[EdgeData]]:
    """Processes edges for graph generation, including prioritization, classification, and final EdgeData creation."""
    (nodes, edges) = create_nodes_and_edges(nodes, company_name)
    number_of_nodes = len(nodes)
    number_of_displayed_edges = get_number_displayed_edges(number_of_nodes)

    edges = _prioritize_edges(
        nodes,
        edges,
        number_of_displayed_edges,
        high_priority_search_space,
        high_priority_atmost_number_edges,
    )

    high_priority_edges = [edge for edge in edges if edge["high_priority"]]
    low_priority_edges = [edge for edge in edges if not edge["high_priority"]]
    print(f"high_priority_edges: {len(high_priority_edges)}")
    print(f"low_priority_edges: {len(low_priority_edges)}")
    (
        processed_edges,
        direction_list,
        edge_label_list,
        classified_count,
    ) = _classify_edges(
        high_priority_edges,
        low_priority_edges,
        company_name,
        embedding_key,
        classify_model_name,
        relation_process,
        get_number_displayed_edges(number_of_nodes),
    )

    final_edge_data_list = _create_final_edge_data(processed_edges)

    return (
        processed_edges,
        direction_list,
        edge_label_list,
        classified_count,
        final_edge_data_list,
    )


def find_cluster_group_for_nodes(
    company_graph_data: CompanyGraphData,
    cluster_method: str = "leiden",
) -> CompanyGraphData:
    """Finds the cluster group for each node."""
    # find the cluster group for each node
    # return the nodes with the cluster group
    nx_graph, _ = get_networkx_graph_from_company_graph_data(company_graph_data)
    if cluster_method == "leiden":
        # clusters = leiden_communities(G, backend="cugraph") # this method needs cugraph backend
        # Step 3.2: Convert NetworkX to igraph
        # G_ig = ig.Graph.TupleList(G.edges(), directed=False)
        G_ig = ig.Graph.TupleList(
            nx_graph.edges(), directed=True, vertex_name_attr="name"
        )

        # Step 3.3: Run Leiden algorithm
        # partition = leidenalg.find_partition(G_ig, leidenalg.CPMVertexPartition) # doesn't seem to work with directed graph
        partition = leidenalg.find_partition(G_ig, leidenalg.RBERVertexPartition)
        # partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition) # for non-directed

        # Step 3.4: Extract communities (as lists of original node names)
        clusters = [
            [G_ig.vs[node]["name"] for node in community] for community in partition
        ]
        # make sure to sort len member of each cluster by len of cluster
        clusters.sort(key=lambda x: len(x), reverse=True)
    nodes = company_graph_data.nodes
    for cluster_id, cluster in enumerate(clusters):
        for node in nodes:
            if node.data.id in cluster:
                node.data.cluster_id = cluster_id

    # update nodes back to company_graph_data
    company_graph_data.nodes = nodes
    return company_graph_data


def generate_graph_elements_for_company(
    company_data: pd.DataFrame,
    company_name: str,
    classify_model_name: str = "gpt-4.1-mini",
    high_priority_search_space: float = 4.0,
    high_priority_atmost_number_edges: int = 3,
    relation_process: str = "oneway_run",
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
        with get_openai_callback() as cb:
            nodes = create_nodes_with_embedding(data_list, company_name, embedding_key)
            (
                processed_edges,
                direction_list,
                edge_label_list,
                classified_count,
                final_edge_data_list,
            ) = _process_edges(
                nodes,
                company_name,
                embedding_key,
                classify_model_name,
                relation_process,
                high_priority_search_space,
                high_priority_atmost_number_edges,
            )

            company_graph_id = f"{company_name}|{embedding_key}|{relation_process}"
            if edge_label_list:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                # convert edge_label_list to pandas dataframe
                edge_label_list_df = pd.DataFrame(edge_label_list)

                with open(
                    f"{dir_path}/edge_label_list_{company_graph_id}.xlsx", "wb"
                ) as f:
                    edge_label_list_df.to_excel(f)
            print(f"direction_list: {direction_list}")
            # raise
            # 5. Process all edges (classified and unclassified) to create EdgeData objects

            print(
                f"number_of_displayed_edges: {get_number_displayed_edges(len(nodes))}"
            )  # Changed to reflect dynamic calculation within _process_edges
            print(f"company_graph_id: {company_graph_id}")
            print(cb)
            thb = cb.total_cost * 35
            print(f"total cost (THB): {thb}")
            company_graph_data = CompanyGraphData(
                nodes=nodes,
                edges=final_edge_data_list,
                number_of_displayed_edges=get_number_displayed_edges(len(nodes)),
            )
            company_graph_data = find_cluster_group_for_nodes(company_graph_data)
            company_graph_datas[company_graph_id] = company_graph_data

    return company_graph_datas


def find_similar_embeddings(
    input_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    top_n: Optional[int] = None,
) -> List[int]:
    """
    Finds the indices of the most similar embeddings in reference_embeddings to the input_embedding.
    Returns indices sorted from most similar to least similar.
    """
    # Reshape the input embedding to a 2D array if it's 1D, for dot product compatibility
    cosine_similarities = np.dot(
        input_embedding.reshape(1, -1), reference_embeddings.T
    ).flatten()

    # Argsort returns indices that would sort an array in ascending order.
    # To get most similar first (highest cosine similarity), we sort in descending order.
    sorted_indices = np.argsort(cosine_similarities)[::-1]

    if top_n is not None:
        return sorted_indices[:top_n].tolist()
    return sorted_indices.tolist()


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
        # find the top n related risk data
        top_n_related_risk_data = find_similar_embeddings(
            embedding, all_embeddings, top_n=top_n
        )
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
                        risk_id=risk_catalog_list[related_risk_data_index].data.risk_id,
                        risk_cat=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_cat,
                        risk_level=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_level,
                        risk_score=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_score,
                        risk_impact=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_impact,
                        risk_likelihood=risk_catalog_list[
                            related_risk_data_index
                        ].data.risk_likelihood,
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


def final_relationship(
    interdependency_type_a_b,
    interdependency_type_b_a,
    direction_a_b,
    direction_b_a,
    rationale_a_b,
    rationale_b_a,
    confidence_a_b,
    confidence_b_a,
):
    priority_interdependency_type_list = [
        "Causal",
        "Correlated",
        "None",
    ]
    priority_direction_list = [
        "Both",
        "A → B",
        "B → A",
        "None",
    ]

    ####### preprocess
    if interdependency_type_b_a in ["Causal"]:
        if direction_b_a == "A → B":
            direction_b_a = "B → A"
        elif direction_b_a == "B → A":
            direction_b_a = "A → B"
    #######
    if interdependency_type_a_b == interdependency_type_b_a:
        final_interdependency_type = interdependency_type_a_b
        if interdependency_type_a_b in ["Causal"]:

            index_direction_a_b = priority_direction_list.index(direction_a_b)
            index_direction_b_a = priority_direction_list.index(direction_b_a)
            if (direction_a_b, direction_b_a) in [
                ("A → B", "B → A"),
                ("B → A", "A → B"),
            ]:
                final_direction = "Both"
                final_rationale = f"{rationale_a_b} / {rationale_b_a}"
                final_confidence = (confidence_a_b + confidence_b_a) / 2
            elif index_direction_a_b < index_direction_b_a:
                final_direction = direction_a_b
                final_rationale = rationale_a_b
                final_confidence = confidence_a_b
            else:
                final_direction = direction_b_a
                final_rationale = rationale_b_a
                final_confidence = confidence_b_a
        else:
            final_direction = "None"
            final_rationale = f"{rationale_a_b} / {rationale_b_a}"
            final_confidence = (confidence_a_b + confidence_b_a) / 2

    else:
        index_interdependency_type_a_b = priority_interdependency_type_list.index(
            interdependency_type_a_b
        )
        index_interdependency_type_b_a = priority_interdependency_type_list.index(
            interdependency_type_b_a
        )
        if index_interdependency_type_a_b < index_interdependency_type_b_a:
            final_interdependency_type = interdependency_type_a_b
            final_direction = direction_a_b
            final_rationale = rationale_a_b
            final_confidence = confidence_a_b
        else:
            final_interdependency_type = interdependency_type_b_a
            final_direction = direction_b_a
            final_rationale = rationale_b_a
            final_confidence = confidence_b_a

    return {
        "interdependency_type": final_interdependency_type,
        "direction": final_direction,
        "rationale": final_rationale,
        "confidence": final_confidence,
    }


def process_two_way_relationships(
    relationship_a_b: Dict, relationship_b_a: Dict
) -> Dict:
    """
    Dummy function to process two-way relationships.
    For now, it always selects relationship A->B as the final data.
    In a real scenario, this would involve merging or reconciling the two relationships.
    """
    # This is where you'd implement the logic to combine/choose between A->B and B->A
    # For example, you might choose the one with higher confidence, or merge rationales.
    # As per the user's request, we'll just return relationship_a_b for now.
    final_relationship_data = final_relationship(
        relationship_a_b["interdependency_type"],
        relationship_b_a["interdependency_type"],
        relationship_a_b["direction"],
        relationship_b_a["direction"],
        relationship_a_b["rationale"],
        relationship_b_a["rationale"],
        relationship_a_b["confidence"],
        relationship_b_a["confidence"],
    )
    return final_relationship_data


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
                risk_impact = risk_data["impact_combined"]
                risk_likelihood = risk_data["likelihood_combined"]
                risk_catalog_reference_datas[timestamp].append(
                    RiskDataWithEmbedding(
                        data=RiskData(
                            id=id,
                            label=label,
                            risk_impact=risk_impact,
                            risk_likelihood=risk_likelihood,
                            **risk_data,
                        ),
                        embedding=risk_data[risk_catalog_embedding_key],
                    )
                )
    for company_name in company_name_list[::-1]:

        if company_name.find("risk_catalog") != -1:
            continue
        company_data = df[df["company"] == company_name]
        for relation_process in ["oneway_run", "twoway_run"]:
            company_graph_data_dict: Dict[str, CompanyGraphData] = (
                generate_graph_elements_for_company(
                    company_data, company_name, relation_process=relation_process
                )
            )

            company_graph_datas.update(company_graph_data_dict)
            # create risk overlay data
            for company_graph_id, company_graph_data in company_graph_data_dict.items():
                overlay_key = (
                    f"risk_catalog_top_n_overlay|{timestamp}|{company_graph_id}"
                )
                risk_catalog_top_n_overlay_data = (
                    generate_risk_catalog_top_n_overlay_data(
                        company_name,
                        risk_catalog_reference_datas[timestamp],
                        company_graph_data,
                        top_n=10,
                    )
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
    # Create and save graph_data_library_no_embedding
    company_graph_datas_no_embedding = {}
    for (
        company_graph_id,
        company_graph_data,
    ) in graph_data_library.company_graph_datas.items():
        company_graph_datas_no_embedding[company_graph_id] = (
            convert_to_no_embedding_company_graph_data(company_graph_data)
        )

    risk_overlay_datas_no_embedding = {}
    for (
        risk_overlay_id,
        risk_overlay_data,
    ) in graph_data_library.risk_overlay_datas.items():
        overlay_data_no_embedding = {}
        for (
            risk_name,
            risk_list_with_embedding,
        ) in risk_overlay_data.overlay_data.items():
            overlay_data_no_embedding[risk_name] = [
                convert_to_no_embedding_risk_data(r) for r in risk_list_with_embedding
            ]
        risk_overlay_datas_no_embedding[risk_overlay_id] = RiskOverlayDataNoEmbedding(
            overlay_data=overlay_data_no_embedding
        )

    risk_catalog_reference_datas_no_embedding = {}
    for (
        timestamp,
        risk_list_with_embedding,
    ) in graph_data_library.risk_catalog_reference_datas.items():
        risk_catalog_reference_datas_no_embedding[timestamp] = [
            convert_to_no_embedding_risk_data(r) for r in risk_list_with_embedding
        ]

    graph_data_library_no_embedding = GraphDataLibraryNoEmbedding(
        company_graph_datas=company_graph_datas_no_embedding,
        risk_overlay_datas=risk_overlay_datas_no_embedding,
        risk_catalog_reference_datas=risk_catalog_reference_datas_no_embedding,
    )

    with open(output_dir / "graph_data_library_no_embedding.pkl", "wb") as f:
        pickle.dump(graph_data_library_no_embedding, f)
    print(
        f"Saved all graphs without embedding to {output_dir / 'graph_data_library_no_embedding.pkl'}"
    )
    with open(output_dir / "graph_data_library_no_embedding_dict.pkl", "wb") as f:
        pickle.dump(graph_data_library_no_embedding.model_dump(), f)
    print(
        f"Saved all graphs without embedding to {output_dir / 'graph_data_library_no_embedding_dict.pkl'}"
    )

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
