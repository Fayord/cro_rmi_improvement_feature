from typing import List
import networkx as nx
import pandas as pd
import numpy as np
from networkx.classes.reportviews import InDegreeView, OutDegreeView

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


def filter_non_arrow_edges2(edges: List[EdgeData]) -> List[EdgeData]:
    new_edges = []

    for edge in edges:
        if edge.interdependency_type == "Causal":
            new_edges.append(edge)
    return new_edges


def get_networkx_graph_from_company_graph_data(
    company_graph_data: CompanyGraphData,
) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    nodes: List[RiskDataWithEmbedding] = company_graph_data.nodes
    edges: List[EdgeData] = company_graph_data.edges
    filter_edges = filter_non_arrow_edges2(edges)
    for edge in filter_edges:
        nx_graph.add_edge(
            edge.source,
            edge.target,
            weight=edge.cosine_similarity,
        )
    in_degree_centrality_dict = nx.in_degree_centrality(nx_graph)
    out_degree_centrality_dict = nx.out_degree_centrality(nx_graph)
    betweenness_dict_weight = nx.betweenness_centrality(nx_graph, weight="weight")
    betweenness_dict_non_weight = nx.betweenness_centrality(nx_graph)
    all_data_list = []  # for convert to dataframe later
    for node in nodes:
        row_data = {
            # "company": company,
            "risk_id": node.data.id,
            "risk_name": node.data.label,
            "risk_level": node.data.risk_level,
            # "in_degree": in_deg,
            "in_degree": nx_graph.in_degree(node.data.id),
            "out_degree": nx_graph.out_degree(node.data.id),
            "in_degree_centrality": in_degree_centrality_dict.get(node.data.id, None),
            "out_degree_centrality": out_degree_centrality_dict.get(node.data.id, None),
            "betweenness_centrality_weight": betweenness_dict_weight.get(
                node.data.id, None
            ),
            "betweenness_centrality_non_weight": betweenness_dict_non_weight.get(
                node.data.id, None
            ),
        }
        all_data_list.append(row_data)

    all_data_df = pd.DataFrame(all_data_list)

    all_data_df["in_degree"] = all_data_df["in_degree"].apply(
        lambda x: np.nan if isinstance(x, InDegreeView) else int(x)
    )
    all_data_df["out_degree"] = all_data_df["out_degree"].apply(
        lambda x: np.nan if isinstance(x, OutDegreeView) else int(x)
    )
    return nx_graph, all_data_df
