import plotly.graph_objects as go
import networkx as nx
import pandas as pd


def create_network(nodes_data, edges_data):
    """
    Create a network graph from nodes and edges data

    Args:
        nodes_data (list): List of dictionaries containing node information
        edges_data (list): List of dictionaries containing edge information with values

    Returns:
        G (nx.Graph): NetworkX graph object
    """
    G = nx.Graph()

    # Add nodes
    for node in nodes_data:
        G.add_node(node["id"], **node.get("attributes", {}))

    # Add edges with values
    for edge in edges_data:
        G.add_edge(edge["source"], edge["target"], weight=edge.get("value", 1))

    return G


def plot_network_graph(G):
    """
    Create an interactive network visualization using Plotly

    Args:
        G (nx.Graph): NetworkX graph object

    Returns:
        fig (go.Figure): Plotly figure object
    """
    # Calculate layout
    pos = nx.spring_layout(G)

    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"Value: {edge[2].get('weight', 1)}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="text",
        text=edge_text,
        mode="lines",
    )

    # Create node trace
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        marker=dict(size=20, color="#1f77b4", line_width=2),
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode='pan',  # Changed from 'drag' to 'pan'
            clickmode='event',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        ),
    )

    # Add modebar buttons
    fig.update_layout(
        modebar=dict(
            bgcolor='rgba(255, 255, 255, 0.8)',
            orientation='v'
        ),
        modebar_add=['drawclosedpath', 'eraseshape']
    )

    return fig


def example_network():
    import string
    import random

    # Create nodes A-Z
    nodes_data = [{"id": letter} for letter in string.ascii_uppercase]

    # Create edges connecting all nodes to each other with random values
    edges_data = []
    for i, node1 in enumerate(string.ascii_uppercase):
        for node2 in string.ascii_uppercase[
            i + 1 :
        ]:  # Start from i+1 to avoid self-loops and duplicates
            edges_data.append(
                {"source": node1, "target": node2, "value": random.randint(1, 100)}
            )

    # Create and plot network
    G = create_network(nodes_data, edges_data)
    fig = plot_network_graph(G)
    return fig


if __name__ == "__main__":
    from dash import Dash, html, dcc
    import dash

    app = Dash(__name__)

    app.layout = html.Div([dcc.Graph(figure=example_network())])

    if __name__ == "__main__":
        app.run(debug=True, port=7070)  # Changed from run_server to run
