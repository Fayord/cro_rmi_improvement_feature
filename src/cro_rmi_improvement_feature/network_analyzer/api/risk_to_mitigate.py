from typing import List, Tuple
from data_processor.create_graph_data_library import (
    CompanyGraphData,
    RiskData,
    RiskDataWithEmbedding,
    EdgeData,
)
import numpy as np
import networkx as nx
import pandas as pd
from networkx.classes.reportviews import InDegreeView, OutDegreeView
from api.schemas import RiskDataWithTags, ExistingRisk
from typing import Dict
from api.utils import (
    filter_non_arrow_edges2,
    get_networkx_graph_from_company_graph_data,
)

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
    existing_risks: List[ExistingRisk],
) -> List[RiskDataWithTags]:
    # remove recommendations_risk_data_list that already have mitigation plan
    for risk_data_with_tags in recommendations_risk_data_list:
        risk_name = risk_data_with_tags.risk_data.risk
        for existing_risk in existing_risks:
            mitigation_plans = existing_risk.mitigation_plans

            if (
                risk_name == existing_risk.risk_name
                and mitigation_plans is not None
                and mitigation_plans != []
            ):
                recommendations_risk_data_list.remove(risk_data_with_tags)
                break
    return recommendations_risk_data_list


def update_tags_risk_data(
    risk_data_dict: Dict[str, List[RiskData]],
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
            recommnend_risk_data_list.append(
                RiskDataWithTags(
                    risk_data=risk_data,
                    is_source_risk=False,
                    is_central_risk=False,
                    is_high_risk=False,
                    is_shared_root_cause=False,
                    is_news_trended=False,
                    is_emerging_risk=False,
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
