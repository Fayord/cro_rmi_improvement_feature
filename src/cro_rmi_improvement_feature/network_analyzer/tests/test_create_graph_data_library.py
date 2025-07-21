import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
from data_processor.create_graph_data_library import (
    create_nodes_and_edges,
    RiskDataWithEmbedding,
    RiskData,
)


def test_similarity_rank_with_mock_data():
    """
    Test that similarity_rank is correctly assigned based on embedding similarity.
    Two cyber risks should be most similar to each other.
    Three operational risks should be similar to each other.
    Cyber risks and operational risks should be dissimilar.
    """

    # Mock data: 2 cyber, 3 operational risks
    # Embeddings designed to simulate similarity/dissimilarity
    nodes = [
        # Cyber Risk 1
        RiskDataWithEmbedding(
            data=RiskData(
                id="cyber_1",
                label="Cyber Risk A",
                risk="Cyber Risk A",
                risk_cat="Cyber",
                risk_desc="Cyber security breach A",
                risk_level=5,
                process=None,
                rootcause=None,
                risk_desc_summary=None,
            ),
            embedding=np.array(
                [1.0, 0.0, 0.0, 0.0]
            ).tolist(),  # Focus on first dimension
        ),
        # Cyber Risk 2 (very similar to Cyber Risk 1)
        RiskDataWithEmbedding(
            data=RiskData(
                id="cyber_2",
                label="Cyber Risk B",
                risk="Cyber Risk B",
                risk_cat="Cyber",
                risk_desc="Cyber security breach B",
                risk_level=5,
                process=None,
                rootcause=None,
                risk_desc_summary=None,
            ),
            embedding=np.array(
                [0.98, 0.0, 0.0, 0.0]
            ).tolist(),  # Slightly different, but still very high similarity
        ),
        # Operational Risk 1
        RiskDataWithEmbedding(
            data=RiskData(
                id="ops_1",
                label="Operational Risk X",
                risk="Operational Risk X",
                risk_cat="Operational",
                risk_desc="Operational inefficiency X",
                risk_level=3,
                process=None,
                rootcause=None,
                risk_desc_summary=None,
            ),
            embedding=np.array(
                [0.0, 1.0, 0.0, 0.0]
            ).tolist(),  # Focus on second dimension, orthogonal to cyber
        ),
        # Operational Risk 2 (similar to Ops 1)
        RiskDataWithEmbedding(
            data=RiskData(
                id="ops_2",
                label="Operational Risk Y",
                risk="Operational Risk Y",
                risk_cat="Operational",
                risk_desc="Operational inefficiency Y",
                risk_level=3,
                process=None,
                rootcause=None,
                risk_desc_summary=None,
            ),
            embedding=np.array([0.0, 0.98, 0.0, 0.0]).tolist(),  # Slightly different
        ),
        # Operational Risk 3 (similar to Ops 1 and Ops 2)
        RiskDataWithEmbedding(
            data=RiskData(
                id="ops_3",
                label="Operational Risk Z",
                risk="Operational Risk Z",
                risk_cat="Operational",
                risk_desc="Operational inefficiency Z",
                risk_level=3,
                process=None,
                rootcause=None,
                risk_desc_summary=None,
            ),
            embedding=np.array([0.0, 0.97, 0.0, 0.0]).tolist(),  # Slightly different
        ),
    ]

    # Call the function under test
    _, edges = create_nodes_and_edges(nodes, "test_company")

    # Extract relevant edges and their similarity ranks
    # We expect 10 edges in total: C1-C2, O1-O2, O1-O3, O2-O3 (within-group) and 6 cross-group

    # Group edges by category pairs
    cyber_cyber_edges = []
    operational_operational_edges = []
    cyber_operational_edges = []

    for edge in edges:
        source_cat = edge["data"]["source_risk_data"].risk_cat
        target_cat = edge["data"]["target_risk_data"].risk_cat
        rank = edge["data"]["similarity_rank"]

        if source_cat == "Cyber" and target_cat == "Cyber":
            cyber_cyber_edges.append(rank)
        elif source_cat == "Operational" and target_cat == "Operational":
            operational_operational_edges.append(rank)
        elif (source_cat == "Cyber" and target_cat == "Operational") or (
            source_cat == "Operational" and target_cat == "Cyber"
        ):
            cyber_operational_edges.append(rank)

    # Assertions
    # 1. Cyber-Cyber edges should have the lowest ranks (most similar)
    assert len(cyber_cyber_edges) == 1
    assert len(operational_operational_edges) == 3
    assert len(cyber_operational_edges) == 6

    # All cyber-cyber edges should have ranks lower than operational-operational edges
    # And operational-operational edges should have ranks lower than cyber-operational edges
    # Get max rank for within-group similarities and min rank for cross-group dissimilarities
    max_cyber_cyber_rank = max(cyber_cyber_edges)
    max_operational_operational_rank = max(operational_operational_edges)
    min_cyber_operational_rank = min(cyber_operational_edges)

    assert max_cyber_cyber_rank < min(
        operational_operational_edges
    ), f"max_cyber_cyber_rank: {max_cyber_cyber_rank}, min(operational_operational_edges): {min(operational_operational_edges)}"
    assert max_operational_operational_rank < min_cyber_operational_rank

    # Also verify that the overall sorting makes sense
    all_ranks = [e["data"]["similarity_rank"] for e in edges]
    # The ranks are 0 to (total_edges - 1). Lower rank means higher similarity.
    # So, the smallest ranks should correspond to the most similar pairs.
    sorted_edges_by_rank = sorted(edges, key=lambda x: x["data"]["similarity_rank"])

    # The top 1 most similar should be cyber-cyber
    assert (
        sorted_edges_by_rank[0]["data"]["source_risk_data"].risk_cat == "Cyber"
        and sorted_edges_by_rank[0]["data"]["target_risk_data"].risk_cat == "Cyber"
    )

    # The next 3 most similar should be operational-operational (after the cyber-cyber one)
    # The rank 0 is C-C
    # The ranks 1, 2, 3 should be O-O (or vice versa depending on exact distance)
    # Let's just check that all O-O edges are ranked before all C-O edges
    all_operational_operational_ranks = sorted(operational_operational_edges)
    all_cyber_operational_ranks = sorted(cyber_operational_edges)

    # The highest rank among operational-operational should be lower than the lowest rank among cyber-operational
    assert all_operational_operational_ranks[-1] < all_cyber_operational_ranks[0]
