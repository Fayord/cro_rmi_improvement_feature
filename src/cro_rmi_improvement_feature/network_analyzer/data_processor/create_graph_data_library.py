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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
from scipy.spatial.distance import pdist, squareform
from pydantic import BaseModel, Field
from data_processor.relation_classifier import classify_relationship


class RiskData(BaseModel):
    id: str = Field(..., description="The id of the risk node.")
    label: str = Field(..., description="Text to display for the risk node.")
    risk: str = Field(..., description="The name of the risk.")
    risk_cat: str = Field(..., description="The category of the risk.")
    risk_desc: str = Field(..., description="The description of the risk.")
    risk_level: int = Field(..., description="The level of the risk.")
    process: Union[List[str], str, None] = Field(
        ..., description="The process of the risk."
    )
    rootcause: Union[List[str], str, None] = Field(
        ..., description="The root cause of the risk."
    )
    process_summary: Optional[str] = Field(
        None, description="The summary of the process."
    )
    rootcause_summary: Optional[str] = Field(
        None, description="The summary of the root cause."
    )
    risk_desc_summary: Optional[str] = Field(
        ..., description="The summary of the risk description."
    )


class RiskDataWithEmbedding(BaseModel):
    data: RiskData = Field(..., description="The risk node data.")
    embedding: List[float] = Field(..., description="The embedding of the risk node.")


class RiskOverlayData(BaseModel):
    overlay_data: Dict[str, List[RiskDataWithEmbedding]] = Field(
        ..., description="A dictionary mapping risk name to list of risk data"
    )


class EdgeData(BaseModel):
    source: str = Field(..., description="The source of the edge. risk_a_data.id")
    target: str = Field(..., description="The target of the edge. risk_b_data.id")
    interdependency_type: Optional[str] = Field(
        None, description="The interdependency type of the edge."
    )
    direction: Optional[str] = Field(None, description="The direction of the edge.")
    rationale: Optional[str] = Field(None, description="The rationale of the edge.")
    confidence: Optional[float] = Field(None, description="The confidence of the edge.")
    risk_a_data: RiskData = Field(..., description="The risk node data of risk a.")
    risk_b_data: RiskData = Field(..., description="The risk node data of risk b.")
    distance: float = Field(..., description="The distance between the two risks.")
    similarity_rank: int = Field(..., description="The similarity rank of the edge.")


class CompanyGraphData(BaseModel):
    """
    Represents the graph data for a specific company and embedding type.
    This is a flexible structure; you can make it more specific
    (e.g., nodes: List[Node], edges: List[Edge]) if needed.
    """

    nodes: List[RiskDataWithEmbedding] = Field(
        ..., description="List of node objects, each with properties."
    )
    edges: List[EdgeData] = Field(
        ...,
        description="List of edge objects, each with source, target, and properties.",
    )
    number_of_displayed_edges: Optional[int] = Field(
        None, description="Number of edges to display in the graph."
    )
    risk_catalog_reference_id: Optional[str] = Field(
        None, description="Version of the risk catalog used for reference."
    )
    overlay_top_n_risks_catalog_id: Optional[str] = Field(
        None, description="Version of the top n risks catalog."
    )
    overlay_news_id_list: Optional[List[str]] = Field(
        None, description="List of news ids to overlay sub graph"
    )


class GraphDataLibrary(BaseModel):
    """
    The main schema representing graph data organized by embedding type -> company -> graph data
    """

    company_graph_datas: Dict[str, CompanyGraphData] = Field(
        ...,
        description="A dictionary mapping company_graph_id str join by '|' (datestamp|company|embedding_source_type) to its CompanyGraphData.",
    )

    risk_overlay_datas: Dict[str, RiskOverlayData] = Field(
        ...,
        description="A dictionary mapping risk_overlay_id str join by '|' (overlay_type|timestamp|embedding_source_type or overlay_name) to its RiskOverlayData.",
    )
    risk_catalog_reference_datas: Dict[str, List[RiskDataWithEmbedding]] = Field(
        ...,
        description="A dictionary mapping timestamp to the risk data.",
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
    nodes: List[Dict[str, Any]],
    company: str,
) -> Tuple[List[RiskDataWithEmbedding], List[Dict[str, Any]]]:
    """Creates nodes and edges for a company."""
    edges = []
    # Store (distance, original_index) to sort distances and then map back to edges
    distance_with_indices = []

    edge_counter = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):

            # distance cosine from embedding
            distance = np.dot(
                nodes[i].embedding,
                nodes[j].embedding,
            )
            distance = distance / (
                np.linalg.norm(nodes[i].embedding) * np.linalg.norm(nodes[j].embedding)
            )

            edges.append(
                {
                    "risk_a_data": nodes[i].data,
                    "risk_b_data": nodes[j].data,
                    "distance": (1 - distance) / 2,  # normalize to 0-1
                }
            )
            distance_with_indices.append((distance, edge_counter))
            edge_counter += 1

    # Sort distances in descending order (higher similarity = higher cosine score)
    # The rank should be assigned such that the most similar (highest cosine score) has the lowest rank (0)
    sorted_distance_with_indices = sorted(
        distance_with_indices, key=lambda x: x[0], reverse=True
    )

    # Assign similarity rank based on the sorted order
    for rank, (distance, original_index) in enumerate(sorted_distance_with_indices):
        edges[original_index]["similarity_rank"] = rank

    return nodes, edges


def generate_graph_elements_for_company(
    company_data: pd.DataFrame,
    company_name: str,
    classify_model_name: str = "gpt-4.1-mini",
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
        nodes, edges = create_nodes_and_edges(nodes, company_name)
        # calculate distance threshold
        number_of_nodes = len(nodes)
        number_of_displayed_edges = 2 * number_of_nodes
        for edge in edges:
            similarity_rank = edge["similarity_rank"]
            risk_a_data: RiskData = edge["risk_a_data"]
            risk_b_data: RiskData = edge["risk_b_data"]
            if similarity_rank < number_of_displayed_edges:
                # do the edge relationship classification
                relationship = classify_relationship(
                    risk_a_data.model_dump(),
                    risk_b_data.model_dump(),
                    classify_model_name,
                )

                if relationship["direction"] in ["A → B", "Both"]:
                    edge["source"] = risk_a_data.id
                    edge["target"] = risk_b_data.id
                else:
                    edge["source"] = risk_b_data.id
                    edge["target"] = risk_a_data.id
            else:
                # do not need to classify the relationship so no need to care who is source and who is target
                # then set a to source and b to target
                edge["source"] = risk_a_data.id
                edge["target"] = risk_b_data.id
                relationship = {
                    "interdependency_type": None,
                    "direction": None,
                    "rationale": None,
                    "confidence": None,
                }
            edge = EdgeData(
                source=edge["source"],
                target=edge["target"],
                interdependency_type=relationship["interdependency_type"],
                direction=relationship["direction"],
                rationale=relationship["rationale"],
                confidence=relationship["confidence"],
                risk_a_data=edge["risk_a_data"],
                risk_b_data=edge["risk_b_data"],
                distance=edge["distance"],
                similarity_rank=edge["similarity_rank"],
            )
        print(f"number_of_displayed_edges: {number_of_displayed_edges}")
        company_graph_id = f"{company_name}|{embedding_key}"
        company_graph_datas[company_graph_id] = CompanyGraphData(
            nodes=nodes,
            edges=edges,
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
    for company_name in company_name_list:
        if company_name.find("risk_catalog") != -1:
            continue
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
