# Standard library imports
import json
from typing import Dict, List, Tuple

# Third-party imports
import pandas as pd
import networkx as nx

# Local imports
from api.schemas import RiskDataWithTags, ExistingRisk
from api.utils import (
    filter_non_arrow_edges2,
    get_networkx_graph_from_company_graph_data,
)
from collections import defaultdict
from data_processor.create_graph_data_library import (
    CompanyGraphData,
    GraphDataLibrary,
    RiskData,
    RiskDataWithEmbedding,
    EdgeData,
)
from api.risk_to_assess import (
    convert_existing_risk_to_risk_data,
    create_company_graph_data,
    save_company_graph_data,
)
from time import time
import uuid

N_TOP = 3


def top_n_with_threshold(
    df: pd.DataFrame, column: str, n: int = 3, threshold: int = 0
) -> pd.DataFrame:
    """
    Keep:
      1) All rows with the highest value (even if count > n), BUT only if highest >= threshold.
      2) Then add rows of the next-highest values while total rows <= n.
      3) Stop before adding any group that would exceed n.

    If max(column) < threshold, return an empty DataFrame.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")

    # Highest must be greater than threshold
    max_val = df[column].max()
    if pd.isna(max_val) or not (max_val >= threshold):
        return df.iloc[0:0]  # empty with same columns

    # Sort and count occurrences of each value (descending by value)
    df_sorted = df.sort_values(by=column, ascending=False)
    counts = df_sorted[column].value_counts().sort_index(ascending=False)

    total = 0
    allowed_values = []

    for i, (value, count) in enumerate(counts.items()):
        if i == 0:
            # Always keep the entire highest group
            allowed_values.append(value)
            total += count
        elif total + count <= n:
            allowed_values.append(value)
            total += count
        else:
            break

    return df_sorted[df_sorted[column].isin(allowed_values)]


def _process_company_graph_data(
    existing_risk_list: List[ExistingRisk], graph_data_library: GraphDataLibrary
) -> Tuple[List[RiskData], CompanyGraphData]:
    """
    Converts existing risks to RiskData, creates company graph data, and saves it.
    """
    company_id = existing_risk_list[0].company_id
    company_graph_id = f"{company_id}|embedding_risk_desc_catalog|oneway_run"

    start_time = time()
    risk_data_list = convert_existing_risk_to_risk_data(
        existing_risk_list,
        do_summarize=True,
    )
    end_time = time()
    print(
        f"Time taken to convert existing risk to risk data and summarize: {end_time - start_time} seconds"
    )

    start_time = time()
    company_graph_data = create_company_graph_data(
        risk_data_list,
        company_name=company_id,
        embedding_key="embedding_risk_desc_catalog",
        classify_model_name="gpt-4.1-mini",
        high_priority_search_space=4.0,
        high_priority_atmost_number_edges=3,
        relation_process="oneway_run",
        graph_data_library=graph_data_library,
    )
    end_time = time()
    print(f"Time taken to create company graph data: {end_time - start_time} seconds")

    start_time = time()
    save_company_graph_data(company_graph_data, company_graph_id, graph_data_library)
    end_time = time()
    print(f"Time taken to save company graph data: {end_time - start_time} seconds")

    return risk_data_list, company_graph_data


def _generate_mitigation_recommendations_with_tags(
    risk_data_list: List[RiskData],
    company_graph_data: CompanyGraphData,
    existing_risk_list: List[ExistingRisk],
) -> List[RiskDataWithTags]:
    """
    Generates mitigation recommendations with tags based on risk data and company graph data.
    """
    is_have_mitigation_plan_mapping_dict = get_is_have_mitigation_plan_mapping_dict(
        existing_risk_list
    )
    risk_name_to_user_ids_mapping_dict = get_risk_name_to_user_ids_mapping_dict(
        existing_risk_list
    )

    high_critical_risks = [risk for risk in risk_data_list if risk.risk_level >= 3]
    source_risks, central_risks = get_source_and_central_risk(company_graph_data)

    risk_data_dict = {
        "is_high_risk": high_critical_risks,
        "is_source_risk": source_risks,
        "is_central_risk": central_risks,
    }
    print(f"source_risks: {len(source_risks)}")
    print(f"central_risks: {len(central_risks)}")

    recommendations_risks: List[RiskDataWithTags] = (
        _add_tags_and_assign_user_ids_to_risk_data(
            risk_data_dict, risk_name_to_user_ids_mapping_dict
        )
    )
    recommendations_risks = remove_existing_risk_with_mitigation_plan(
        recommendations_risks, is_have_mitigation_plan_mapping_dict
    )
    return recommendations_risks


def get_source_and_central_risk(
    company_graph_data: CompanyGraphData, top_n: int = N_TOP
) -> Tuple[List[RiskData], List[RiskData]]:
    # convert company_graph_data to nx.DiGraph() and dataframe
    # use it to find the source and central risk with existing code from notebook
    nx_graph, all_data_df = get_networkx_graph_from_company_graph_data(
        company_graph_data
    )
    # find the source and central risk with existing code from notebook
    central_risks_df = top_n_with_threshold(
        all_data_df,
        "betweenness_centrality_non_weight",
        n=top_n,
        threshold=0.001,  # to prevent empty central risks
    )
    source_risks_df = top_n_with_threshold(
        all_data_df,
        "out_degree",
        n=top_n,
        threshold=2,  # to prevent all risks is source risks in small graph
    )
    central_risks: List[RiskData] = []
    source_risks: List[RiskData] = []
    # loop in nodes and find risk_id == node.data.id
    for node in company_graph_data.nodes:
        if node.data.id in central_risks_df["risk_id"].values:
            central_risks.append(node.data)
        if node.data.id in source_risks_df["risk_id"].values:
            source_risks.append(node.data)

    return source_risks, central_risks


def remove_existing_risk_with_mitigation_plan(
    recommendations_risk_data_list: List[RiskDataWithTags],
    is_have_mitigation_plan_mapping_dict: Dict[Tuple[str, str], bool],
) -> List[RiskDataWithTags]:
    # Create a mapping for efficient lookup of RiskDataWithTags objects by risk_name
    risk_name_to_recommendation_map = {
        r.risk_data.risk: r for r in recommendations_risk_data_list
    }

    # Iterate through the mitigation plan dictionary and remove user_ids
    for (
        risk_name,
        user_id,
    ), has_mitigation_plan in is_have_mitigation_plan_mapping_dict.items():
        if has_mitigation_plan and risk_name in risk_name_to_recommendation_map:
            recommendation = risk_name_to_recommendation_map[risk_name]
            # Ensure the user_id exists in the list before trying to remove it
            if user_id in recommendation.user_ids:
                recommendation.user_ids.remove(user_id)

    # Filter out recommendations where user_ids list is now empty
    filtered_recommendations = [r for r in recommendations_risk_data_list if r.user_ids]
    return filtered_recommendations


def _add_tags_and_assign_user_ids_to_risk_data(
    risk_data_dict: Dict[str, List[RiskData]],
    risk_name_to_user_ids_mapping_dict: Dict[str, List[str]],
) -> List[RiskDataWithTags]:
    recommnend_risk_data_list: List[RiskDataWithTags] = []
    # check is dict keys is in Literal tags
    for key in risk_data_dict.keys():
        if key not in [
            "is_central_risk",
            "is_source_risk",
            "is_high_risk",
            "is_shared_root_cause",
            "is_news_trended",
            "is_emerging_risk",
        ]:
            raise ValueError(f"Invalid key: {key}")
    # loop all List risk in data

    for risk_data_list in risk_data_dict.values():
        for risk_data in risk_data_list:
            user_ids = risk_name_to_user_ids_mapping_dict[risk_data.risk]
            recommnend_risk_data_list.append(
                RiskDataWithTags(
                    risk_data=risk_data,
                    user_ids=user_ids,
                )
            )
    # then loop in each key in risk_data_dict and update the tags
    for key in risk_data_dict.keys():
        for risk_data in risk_data_dict[key]:
            for recommnend_risk_data in recommnend_risk_data_list:
                if recommnend_risk_data.risk_data.risk == risk_data.risk:
                    setattr(recommnend_risk_data, key, True)
                    break

    return recommnend_risk_data_list


def get_risk_name_to_user_ids_mapping_dict(
    existing_risks: List[ExistingRisk],
) -> Dict[str, List[str]]:
    risk_name_to_user_ids_mapping_dict = defaultdict(list)
    for existing_risk in existing_risks:
        risk_name = existing_risk.risk_name
        user_id = existing_risk.user_id
        risk_name_to_user_ids_mapping_dict[risk_name].append(user_id)
    return risk_name_to_user_ids_mapping_dict


def get_is_have_mitigation_plan_mapping_dict(
    existing_risks: List[ExistingRisk],
) -> Dict[Tuple[str, str], bool]:
    # loop in existing_risks and get the risk_name and user_id
    is_have_mitigation_plan_mapping_dict = {}
    for existing_risk in existing_risks:
        risk_name = existing_risk.risk_name
        user_id = existing_risk.user_id
        is_have_mitigation_plan = (
            existing_risk.mitigation_plans is not None
            and existing_risk.mitigation_plans != []
        )
        is_have_mitigation_plan_mapping_dict[(risk_name, user_id)] = (
            is_have_mitigation_plan
        )
    return is_have_mitigation_plan_mapping_dict
