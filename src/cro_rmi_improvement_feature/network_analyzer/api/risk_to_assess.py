"""
Module: risk_to_assess

This module provides core logic for the risk recommendation FastAPI service (main.py).
It is responsible for:

1. Receiving all risk data for a given company.
2. Saving the data and checking for updates; updating records if necessary.
3. Filtering for "interested" risks (currently: high-level risks).
4. Selecting the top N overlay risks and identifying those not already present in the existing risk set.

Intended for import and use in main.py to power the /recommend_risk_to_assess endpoint.
"""

from api.schemas import (
    RiskRecommendationAssessmentResponse,
    RiskDataWithTags,
    ExistingRisk,
)

from typing import List, Set, Optional, Dict, Tuple
import pickle
import sys
import os
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


from core.embedding_providers import (
    OpenAIEmbeddingProvider,
)
from core.models import Process, RootCause
from config.settings import NUMBER_OF_WORKERS
from data_processor.create_graph_data_library import (
    GraphDataLibrary,
    CompanyGraphData,
    RiskDataWithEmbedding,
    RiskData,
    EdgeData,
    RiskOverlayData,
    find_similar_embeddings,
    _process_edges,
    create_nodes_and_edges,
    _prioritize_edges,
    _classify_edges,
    _create_final_edge_data,
    get_number_displayed_edges,
)
from data_processor.summarize_data import summarize_risk


def _save_or_update_risk_data(
    company_id: str, risk_data: List[ExistingRisk]
) -> None: ...


def get_graph_data_library_path() -> str:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    GRAPH_DATA_PATH = os.path.join(
        dir_path, "..", "data", "graph", "graph_data_library.pkl"
    )
    return GRAPH_DATA_PATH


def load_graph_data_library() -> GraphDataLibrary:
    """Loads the graph data library from the data directory."""
    graph_data_library_path = get_graph_data_library_path()
    with open(graph_data_library_path, "rb") as f:
        return pickle.load(f)


def filter_interested_risk(
    risk_data_list: List[RiskData],
) -> List[RiskData]:
    """Filters for interested risks."""
    # first get risk that has risk_level >= 3
    interested_risks = [risk for risk in risk_data_list if risk.risk_level >= 3]
    # if interested_risks has 0 member then pick risk that have highest risk_score with highest risk_impact
    if len(interested_risks) == 0:
        highest_risk_score_risk_list = [
            risk
            for risk in risk_data_list
            if risk.risk_score == max(risk.risk_score for risk in risk_data_list)
        ]
        # then get highest risk_impact among highest_risk_score_risk_list
        highest_risk_impact_among_highest_risk_score = max(
            risk.risk_impact for risk in highest_risk_score_risk_list
        )
        highest_risk_score_with_highest_risk_impact_risk_list = [
            risk
            for risk in highest_risk_score_risk_list
            if risk.risk_impact == highest_risk_impact_among_highest_risk_score
        ]
        interested_risks.extend(highest_risk_score_with_highest_risk_impact_risk_list)
    return interested_risks


def create_embedding_text_raw_user_data(risk: RiskData) -> str:
    processes = []
    root_causes = []
    for process in risk.process:
        processes.append(f"{process.name}: {process.description}")
    for root_cause in risk.rootcause:
        root_causes.append(f"{root_cause.name}: {root_cause.description}")
    processes_str = ",  ".join(processes)
    root_causes_str = ", ".join(root_causes)
    return f"Risk: {risk.risk} | Description: {risk.risk_desc} | Processes: {processes_str} | Root Causes: {root_causes_str}"


def create_embedding_text_risk_desc_catalog(
    risk: RiskData, timestamp: str, graph_data_library: GraphDataLibrary
) -> str:

    risk_catalog_reference_data: List[RiskDataWithEmbedding] = (
        graph_data_library.risk_catalog_reference_datas[timestamp]
    )
    # find risk catalog that have risk name same as risk.risk
    text_parts = []
    text_parts.append(f"Risk: {risk.risk}")
    text_parts.append(f"Description: {risk.risk_desc}")
    for risk_cat in risk_catalog_reference_data:
        if risk_cat.data.risk == risk.risk:
            text_parts.append(f"Catalog Description: {risk_cat.data.risk_desc}")
            break
    return " | ".join(text_parts)


def create_embedding(text: str, use_cache: bool = True) -> List[float]:
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-large")

    def get_openai_large_embedding(text: str) -> List[float]:
        """Get embedding from OpenAI large model."""
        if not text or text.strip() == "":
            return None
        try:
            embedding = provider.get_embedding(text, use_cache=use_cache)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting embedding for text: {e}")
            return None

    return get_openai_large_embedding(text)


def create_company_graph_data(
    risk_data_list: List[RiskData],
    graph_data_library: GraphDataLibrary,
    company_name: str = "User Company",
    embedding_key: str = "embedding_risk_desc_catalog",
    classify_model_name: str = "gpt-4.1-mini",
    high_priority_search_space: float = 4.0,
    high_priority_atmost_number_edges: int = 3,
    relation_process: str = "oneway_run",
) -> CompanyGraphData:
    embedding_text_list = [
        create_embedding_text_risk_desc_catalog(risk, "20250513", graph_data_library)
        for risk in risk_data_list
    ]
    embedding_risk_data_list = []
    with ThreadPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
        embedding_risk_data_list = list(
            executor.map(create_embedding, embedding_text_list)
        )
    nodes = [
        RiskDataWithEmbedding(data=risk, embedding=embedding_risk_data_list[i])
        for i, risk in enumerate(risk_data_list)
    ]

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

    # The original function had `edges = []` here.
    # Now, `final_edge_data_list` contains the processed edges.
    # print(f"direction_list: {direction_list}") # Commented out, as this is for debugging/logging
    return CompanyGraphData(
        nodes=nodes,
        edges=final_edge_data_list,
        number_of_displayed_edges=get_number_displayed_edges(len(nodes)),
    )


def update_company_graph_data(
    prev_company_graph_data: Optional[CompanyGraphData],
    new_company_graph_data: CompanyGraphData,
) -> CompanyGraphData:
    if prev_company_graph_data is None:
        return new_company_graph_data
    else:
        ...


def update_risk_data_list(
    prev_risk_data_list: List[RiskData],
    new_risk_data_list: List[RiskData],
) -> List[RiskData]:
    # union
    ...


def save_company_graph_data(
    company_graph_data: CompanyGraphData,
    company_graph_id: str,
    graph_data_library: GraphDataLibrary,
) -> None:
    # update graph data library
    # save individual company graph data
    graph_data_library_path = get_graph_data_library_path()
    graph_data_library_folder_path = os.path.dirname(graph_data_library_path)
    individual_company_graph_data_path = os.path.join(
        graph_data_library_folder_path,
        "company_graph_data-" + company_graph_id + ".pkl",
    )
    with open(individual_company_graph_data_path, "wb") as f:
        pickle.dump(company_graph_data, f)
    # update graph data library
    graph_data_library.company_graph_datas[company_graph_id] = company_graph_data
    # save graph data library
    with open(graph_data_library_path, "wb") as f:
        pickle.dump(graph_data_library, f)


def recommend_risk_to_assesses(
    risk_data_list: List[RiskData],
    year_quarter: str,
    graph_data_library: GraphDataLibrary,
) -> List[RiskData]:
    year_quarter_to_timestamp_dict = {
        "2024-Q4": "20250513",
        "2025-Q1": "20250513",
        "2025-Q2": "20250513",
        "2025-Q3": "20250513",
    }
    timestamp = year_quarter_to_timestamp_dict[year_quarter]
    start_time = time.perf_counter()
    all_recommended_risks: List[RiskData] = []

    end_load_graph_data_library_time = time.perf_counter()
    print(
        f"Time to load graph data library: {end_load_graph_data_library_time - start_time:.4f} seconds"
    )

    existing_risk_name_set = {risk.risk for risk in risk_data_list}
    interested_risks = filter_interested_risk(risk_data_list)
    end_filter_interested_risk_time = time.perf_counter()
    print(
        f"Time to filter interested risks: {end_filter_interested_risk_time - end_load_graph_data_library_time:.4f} seconds"
    )

    risk_catalog_reference_data = graph_data_library.risk_catalog_reference_datas[
        timestamp
    ]
    embedding_risk_catalog_reference_data = [
        risk.embedding for risk in risk_catalog_reference_data
    ]
    end_get_reference_data_time = time.perf_counter()
    print(
        f"Time to get risk catalog reference data: {end_get_reference_data_time - end_filter_interested_risk_time:.4f} seconds"
    )

    embedding_text_list = [
        create_embedding_text_raw_user_data(risk) for risk in interested_risks
    ]
    # Parallelize embedding creation
    embedding_risk_data_list = []
    with ThreadPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
        embedding_risk_data_list = list(
            executor.map(create_embedding, embedding_text_list)
        )

    end_create_embeddings_time = time.perf_counter()
    print(
        f"Time to create embeddings for interested risks: {end_create_embeddings_time - end_get_reference_data_time:.4f} seconds"
    )

    for embedding_risk_data in embedding_risk_data_list:
        embedding_risk_data = np.array(embedding_risk_data)
        embedding_risk_catalog_reference_data = np.array(
            embedding_risk_catalog_reference_data
        )
        similar_reference_risk_indices = find_similar_embeddings(
            embedding_risk_data, embedding_risk_catalog_reference_data, top_n=10
        )
        similar_risk_data_list: List[RiskData] = [
            risk_catalog_reference_data[index].data
            for index in similar_reference_risk_indices
        ]
        similar_risk_data_list = [
            risk
            for risk in similar_risk_data_list
            if risk.risk not in existing_risk_name_set
        ]
        all_recommended_risks.extend(similar_risk_data_list)
    end_find_similar_risks_time = time.perf_counter()
    print(
        f"Time to find and filter similar risks: {end_find_similar_risks_time - end_create_embeddings_time:.4f} seconds"
    )

    total_time = time.perf_counter() - start_time
    print(f"Total time for recommend_risk_to_assesses: {total_time:.4f} seconds")
    return all_recommended_risks


def convert_existing_risk_to_risk_data(
    existing_risk_list: List[ExistingRisk],
    do_summarize: bool = False,
) -> List[RiskData]:
    risk_data_list = []
    for risk in existing_risk_list:
        processes: List[Process] = [
            Process(id=process.id, name=process.name, description=process.description)
            for process in risk.processes
        ]
        root_causes: List[RootCause] = [
            RootCause(
                id=root_cause.id,
                name=root_cause.name,
                description=root_cause.description,
            )
            for root_cause in risk.root_causes
        ]
        processes_str = "\n".join(
            [f"{process.name}: {process.description}" for process in processes]
        )
        root_causes_str = "\n".join(
            [
                f"{root_cause.name}: {root_cause.description}"
                for root_cause in root_causes
            ]
        )
        risk_data = {
            "risk": risk.risk_name,
            "risk_desc": risk.risk_description,
            "rootcause_data": root_causes_str,
            "process_data": processes_str,
        }

        # Use actual LLM to create summaries
        summaries = summarize_risk(risk_data)
        risk_data_list.append(
            RiskData(
                id=risk.risk_id,
                label=risk.risk_name,
                risk=risk.risk_name,
                risk_id=risk.risk_id,
                risk_cat=risk.risk_category,
                risk_level=risk.score.risk_level,
                risk_score=risk.score.score,
                risk_impact=risk.score.impact,
                risk_likelihood=risk.score.likelihood,
                process=processes,
                risk_desc=risk.risk_description,
                rootcause=root_causes,
                process_summary=summaries.get("process_summary", ""),
                rootcause_summary=summaries.get("rootcause_summary", ""),
                risk_desc_summary=summaries.get("risk_desc_summary", ""),
            )
        )
    return risk_data_list


if __name__ == "__main__":
    import json
    import os

    dir_path = os.path.dirname(os.path.abspath(__file__))
    example_request_path = os.path.join(dir_path, "example_request.json")
    with open(example_request_path, "r") as f:
        request_data = json.load(f)
        existing_risk_list = request_data["existing_risks"]
        # convert existing_risk_list to ExistingRisk
        existing_risk_list = [ExistingRisk(**risk) for risk in existing_risk_list]
    risk_data_list = convert_existing_risk_to_risk_data(existing_risk_list)
    print(risk_data_list[:3])
