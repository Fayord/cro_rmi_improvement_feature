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

from schemas import (
    RiskRecommendationAssessmentResponse,
    RiskDataWithTags,
    ExistingRisk,
)

from typing import List, Set
import pickle
import sys
import os
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
sys.path.append("../../find_similar_risk")

from embedding_providers import (  # type: ignore
    OpenAIEmbeddingProvider,
)
from core.models import Process, RootCause
from data_processor.create_graph_data_library import (
    GraphDataLibrary,
    CompanyGraphData,
    RiskDataWithEmbedding,
    RiskData,
    EdgeData,
    RiskOverlayData,
    find_similar_embeddings,
)


def _save_or_update_risk_data(
    company_id: str, risk_data: List[ExistingRisk]
) -> None: ...


def load_graph_data_library() -> GraphDataLibrary:
    """Loads the graph data library from the data directory."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    GRAPH_DATA_PATH = os.path.join(
        dir_path, "..", "data", "graph", "graph_data_library.pkl"
    )
    with open(GRAPH_DATA_PATH, "rb") as f:
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


def recommend_risk_to_assesses_old(
    company_id: str,
    assessment_risks: List[ExistingRisk],
    assessment_version: str,
    timestamp: str = "20250513",
) -> List[RiskData]:
    # 1. Receiving all risk data for a given company. (Handled by input `risk_data`)

    # 2. Save and update risk data
    _save_or_update_risk_data(company_id, assessment_risks)

    all_recommended_risks: List[RiskData] = []
    # load it from graph data library
    graph_data_library = load_graph_data_library()
    # check if company_id is in graph_data_library.company_graph_datas
    if company_id not in graph_data_library.company_graph_datas:
        raise ValueError(f"Company {company_id} not found in graph data library")
    company_graph_data = graph_data_library.company_graph_datas[company_id]
    existing_risk_name_set = {risk.data.risk for risk in company_graph_data.nodes}
    interested_risks = filter_interested_risk(company_graph_data)
    overlay_top_n_risk_id = f"risk_catalog_top_n_overlay|{timestamp}|{company_id}"
    overlay_top_n_graph_data = graph_data_library.risk_overlay_datas[
        overlay_top_n_risk_id
    ]
    overlay_data = overlay_top_n_graph_data.overlay_data
    for risk in interested_risks:
        risk_name = risk.risk
        top_n_overlay_risks = overlay_data[risk_name]
        # filter out the risks that are already in the existing_risk_name_set
        top_n_overlay_risks = [
            risk.data
            for risk in top_n_overlay_risks
            if risk.data.risk not in existing_risk_name_set
        ]
        all_recommended_risks.extend(top_n_overlay_risks)
    return all_recommended_risks


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


def recommend_risk_to_assesses(
    risk_data_list: List[RiskData],
    timestamp: str = "20250513",
) -> List[RiskData]:
    start_time = time.perf_counter()
    all_recommended_risks: List[RiskData] = []
    graph_data_library: GraphDataLibrary = load_graph_data_library()
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
    with ThreadPoolExecutor(max_workers=32) as executor:
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
        risk_data_list.append(
            RiskData(
                id=risk.risk_id,
                label=risk.risk_name,
                risk=risk.risk_name,
                risk_cat=risk.risk_category,
                risk_level=risk.score.risk_level,
                risk_score=risk.score.score,
                risk_impact=risk.score.impact,
                risk_likelihood=risk.score.likelihood,
                process=processes,
                risk_desc=risk.risk_description,
                rootcause=root_causes,
                process_summary="",
                rootcause_summary="",
                risk_desc_summary="",
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
