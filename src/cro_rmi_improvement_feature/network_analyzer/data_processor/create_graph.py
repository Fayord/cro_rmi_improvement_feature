#!/usr/bin/env python3
"""
Graph creation module for the Risk Network Visualization System.

This module creates graphs from processed data with embeddings.
"""

import os
import json
import pandas as pd
import networkx as nx
from typing import Dict, List, Any
from pathlib import Path


def load_embeddings_data(data_path: str) -> pd.DataFrame:
    """
    Load embeddings data from JSON or pickle format.

    Args:
        data_path: Path to the embeddings data file

    Returns:
        DataFrame with embeddings data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Embeddings data file not found: {data_path}")

    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif data_path.endswith(".pkl"):
        df = pd.read_pickle(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    print(f"Loaded embeddings data: {df.shape}")
    return df


def create_risk_network(df: pd.DataFrame) -> nx.Graph:
    """
    Create a NetworkX graph from risk data.

    Args:
        df: DataFrame with risk data and embeddings

    Returns:
        NetworkX graph
    """
    print("Creating risk network graph...")

    # Create an empty graph
    G = nx.Graph()

    # Add nodes (risks)
    for idx, row in df.iterrows():
        risk_name = row.get("risk", f"Risk_{idx}")
        G.add_node(
            risk_name,
            risk_desc=row.get("risk_desc", ""),
            company=row.get("company", ""),
            risk_level=row.get("risk_level", 0),
            risk_desc_summary=row.get("risk_desc_summary", ""),
            rootcause_summary=row.get("rootcause_summary", ""),
            process_summary=row.get("process_summary", ""),
        )

    print(f"Created graph with {G.number_of_nodes()} nodes")
    return G


def save_graph(G: nx.Graph, output_path: str):
    """
    Save graph to GraphML format.

    Args:
        G: NetworkX graph
        output_path: Path to save the graph file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as GraphML
    nx.write_graphml(G, output_path)
    print(f"Graph saved to: {output_path}")

    # Print graph statistics
    print(f"Graph statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")


def process_graph_creation(
    embeddings_data_path: str, graph_output_path: str
) -> nx.Graph:
    """
    Process graph creation from embeddings data.

    Args:
        embeddings_data_path: Path to the embeddings data file
        graph_output_path: Path to save the graph file

    Returns:
        NetworkX graph
    """
    print("Starting graph creation process...")

    # Load embeddings data
    print("Loading embeddings data...")
    df = load_embeddings_data(embeddings_data_path)

    # Create graph
    print("Creating risk network graph...")
    G = create_risk_network(df)

    # Save graph
    print("Saving graph...")
    save_graph(G, graph_output_path)

    print("Graph creation process completed!")
    return G


def main():
    """
    Example usage of the graph creation module.
    """
    # Get the directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Define paths
    data_type = "riskview"

    # Try to load from sample data first, then full data
    sample_embeddings_path = os.path.join(
        dir_path, "../data/embeddings/", f"{data_type}_sample_data_with_embeddings.json"
    )
    full_embeddings_path = os.path.join(
        dir_path, "../data/embeddings/", f"{data_type}_data_with_embeddings.pkl"
    )

    # Choose which file to use
    if os.path.exists(sample_embeddings_path):
        embeddings_data_path = sample_embeddings_path
        print("Using sample embeddings data")
    elif os.path.exists(full_embeddings_path):
        embeddings_data_path = full_embeddings_path
        print("Using full embeddings data")
    else:
        print("No embeddings data found. Please run embedding_process.py first.")
        return

    graph_output_path = os.path.join(
        dir_path, "../data/graphs/", f"{data_type}_network.graphml"
    )

    try:
        # Process graph creation
        G = process_graph_creation(embeddings_data_path, graph_output_path)

        print(f"Processing completed! Created graph with {G.number_of_nodes()} nodes")
        print(f"Graph saved to: {graph_output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run embedding_process.py first to create embeddings data.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
