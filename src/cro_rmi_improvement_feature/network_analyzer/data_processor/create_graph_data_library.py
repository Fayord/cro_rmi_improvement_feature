#!/usr/bin/env python3


import json
import itertools
import os
import pickle
from pathlib import Path
from turtle import position
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from pydantic import BaseModel, Field


class RiskData(BaseModel):
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


class RiskNodeData(RiskData):
    id: str = Field(..., description="The id of the risk node.")
    label: str = Field(..., description="The label of the risk node.")


class RiskNodeDataWithPositionAndEmbedding(BaseModel):
    data: RiskNodeData = Field(..., description="The risk node data.")
    position: Dict[str, Any] = Field(..., description="The position of the risk node.")
    embedding: List[float] = Field(..., description="The embedding of the risk node.")


class RiskOverlayData(BaseModel):
    overlay_data: Dict[str, List[RiskNodeDataWithPositionAndEmbedding]] = Field(
        ..., description="A dictionary mapping risk name to list of risk data"
    )


class CompanyGraphData(BaseModel):
    """
    Represents the graph data for a specific company and embedding type.
    This is a flexible structure; you can make it more specific
    (e.g., nodes: List[Node], edges: List[Edge]) if needed.
    """

    nodes: List[RiskNodeDataWithPositionAndEmbedding] = Field(
        ..., description="List of node objects, each with properties."
    )
    edges: List[Dict[str, Any]] = Field(
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
    risk_catalog_reference_datas: Dict[str, List[RiskData]] = Field(
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


def generate_node_id(company: str, idx: int) -> str:
    """Generates a unique node id for a risk data."""
    return f"risk_{company.replace(' ', '_')}_{idx}"


def create_nodes_with_embedding(
    data_list: List[Dict[str, Any]], company: str, embedding_key: str
) -> List[RiskNodeDataWithPositionAndEmbedding]:
    """Creates a list of node dictionaries for Cytoscape."""
    nodes = []
    for idx, data in enumerate(data_list):
        node_id = generate_node_id(company, idx)
        nodes.append(
            RiskNodeDataWithPositionAndEmbedding(
                data=RiskNodeData(
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
                position={
                    "x": 0,
                    "y": 0,
                },
                embedding=data[embedding_key],
            ),
        )

    return nodes


def create_nodes_and_edges(
    nodes: List[Dict[str, Any]],
    company: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Creates nodes and edges for a company."""
    # create fully connected edges for each node
    # I want distance to be  1-distance so that smaller distance means more similar
    edges = []
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
                    "data": {
                        "source": f"risk_{company}_{i}",
                        "target": f"risk_{company}_{j}",
                        "source_risk_data": nodes[i].data,
                        "target_risk_data": nodes[j].data,
                        "distance": (1 - distance) / 2,  # normalize to 0-1
                    }
                }
            )

    return nodes, edges


def generate_graph_elements_for_company(
    company_data: pd.DataFrame, company_name: str
) -> CompanyGraphData:
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
        print(f"number_of_displayed_edges: {number_of_displayed_edges}")
        company_graph_id = f"{company_name}|{embedding_key}"
        company_graph_datas[company_graph_id] = CompanyGraphData(
            nodes=nodes,
            edges=edges,
            number_of_displayed_edges=number_of_displayed_edges,
        )

    return company_graph_datas


def generate_risk_catalog_top_n_overlay_data(
    risk_catalog_list: List[Dict[str, Any]],
    top_n: int = 10,
) -> Dict[str, List[RiskNodeData]]:
    """Generates overlay data for the risk catalog.
    Dict[str, List[RiskNodeData]] where str is the risk name and List[RiskNodeData] is the top n relatedrisk data
    """
    risk_catalog_top_n_overlay_data = {}
    # Ensure all embeddings are numerical lists/arrays
    all_embeddings = np.array(
        [risk_data["embedding_raw_user_data"] for risk_data in risk_catalog_list]
    )
    for risk_data in risk_catalog_list:
        print(f"risk_data: {risk_data.keys()}")
        # use this embedding_text_raw_user_data as embbedding data to find the top n related risk data
        # use cosine similarity to find the top n related risk data
        # Use the actual numerical embedding, not the text used to generate it
        embedding = np.array(risk_data["embedding_raw_user_data"])
        # calculate the cosine similarity between the embedding and the risk_catalog_list
        # Reshape the embedding to a 2D array if it's 1D, for dot product compatibility
        cosine_similarities = np.dot(
            embedding.reshape(1, -1), all_embeddings.T
        ).flatten()
        # find the top n related risk data
        top_n_related_risk_data = np.argsort(cosine_similarities)[-top_n:]
        risk_catalog_top_n_overlay_data[risk_data["risk"]] = []
        for related_risk_data_index in top_n_related_risk_data:
            id = f"risk_{risk_data['company']}_{related_risk_data_index}"
            label = risk_catalog_list[related_risk_data_index]["risk"]
            risk_catalog_top_n_overlay_data[risk_data["risk"]].append(
                RiskNodeDataWithPositionAndEmbedding(
                    data=RiskNodeData(
                        id=id,
                        label=label,
                        **risk_catalog_list[related_risk_data_index],
                    ),
                    position={
                        "x": 0,
                        "y": 0,
                    },
                    embedding=risk_catalog_list[related_risk_data_index][
                        "embedding_raw_user_data"
                    ],
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
    risk_catalog_reference_datas: Dict[str, List[RiskData]] = {}
    company_name_list = df["company"].unique()
    for company_name in company_name_list:
        company_data = df[df["company"] == company_name]
        if company_name.find("risk_catalog") != -1:
            # overlay_id (overlay_type|timestamp|embedding_source_type or overlay_name)
            # save risk catalog data to risk_catalog_for_reference
            # and create an overlay data for the risk catalog
            timestamp = company_name.split("-")[-1]
            if timestamp not in risk_catalog_reference_datas:
                risk_catalog_reference_datas[timestamp] = []
                risk_catalog_list = company_data.to_dict(orient="records")
                for risk_data in risk_catalog_list:
                    risk_catalog_reference_datas[timestamp].append(
                        RiskData(**risk_data)
                    )
                embedding_key = "embedding_all_risk_catalog_data"
                risk_overlay_keys = (
                    f"risk_catalog_top_n_overlay|{timestamp}|{embedding_key}"
                )
                risk_catalog_top_n_overlay_data = (
                    generate_risk_catalog_top_n_overlay_data(risk_catalog_list)
                )
                risk_overlay_datas[risk_overlay_keys] = RiskOverlayData(
                    overlay_data=risk_catalog_top_n_overlay_data,
                )

        else:
            company_graph_data = generate_graph_elements_for_company(
                company_data, company_name
            )
            company_graph_datas.update(company_graph_data)
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
