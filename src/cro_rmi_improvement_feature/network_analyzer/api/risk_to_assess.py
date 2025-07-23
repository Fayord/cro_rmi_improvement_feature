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
    RiskRecommendationRequest,
    RiskRecommendationResponse,
    RecommendedRisk,
    ExistingRisk,
)
from typing import List, Set
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
from data_processor.create_graph_data_library import (
    GraphDataLibrary,
    CompanyGraphData,
    RiskDataWithEmbedding,
    RiskData,
    EdgeData,
    RiskOverlayData,
)


def _save_or_update_risk_data(
    company_id: str, risk_data: List[RiskRecommendationRequest]
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
    company_graph_data: CompanyGraphData,
) -> List[RiskDataWithEmbedding]:
    """Filters for interested risks."""
    return [risk for risk in company_graph_data.nodes if risk.data.risk_level >= 3]


def recommend_risk_to_assesses(
    company_id: str,
    risk_data: RiskRecommendationRequest,
    timestamp: str = "20250513",
) -> List[RecommendedRisk]:
    # 1. Receiving all risk data for a given company. (Handled by input `risk_data`)

    # 2. Save and update risk data
    _save_or_update_risk_data(company_id, risk_data.existing_risks)

    all_recommended_risks: List[RecommendedRisk] = []
    # load it from graph data library
    graph_data_library = load_graph_data_library()
    company_graph_data = graph_data_library.company_graph_datas[company_id]
    existing_risk_name_set = {risk.data.risk for risk in company_graph_data.nodes}
    interested_risks = filter_interested_risk(company_graph_data)
    overlay_top_n_risk_id = f"risk_catalog_top_n_overlay|{timestamp}|{company_id}"
    overlay_top_n_graph_data = graph_data_library.risk_overlay_datas[
        overlay_top_n_risk_id
    ]
    overlay_data = overlay_top_n_graph_data.overlay_data
    for risk in interested_risks:
        risk_name = risk.data.risk
        top_n_overlay_risks = overlay_data[risk_name]
        # filter out the risks that are already in the existing_risk_name_set
        top_n_overlay_risks = [
            risk.data
            for risk in top_n_overlay_risks
            if risk.data.risk not in existing_risk_name_set
        ]
        recommended_risks = {
            "risk_id": risk.data.id,
            "risk_name": risk.data.risk,
            "recommend_risk_list": top_n_overlay_risks,
        }
        all_recommended_risks.append(recommended_risks)
    return all_recommended_risks
