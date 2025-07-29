"""
Main Dash application entry point.
"""

# Standard library imports
import json
import pickle
from pathlib import Path
import math
import random
from collections import Counter
import argparse

# Third-party imports
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
# Local imports
from data_processor.create_graph_data_library import (
    GraphDataLibrary,
    CompanyGraphData,
    RiskDataWithEmbedding,
    RiskData,
    EdgeData,
    RiskOverlayData,
)
from default_value import (  # Import needed constants
    EDGE_SIZE_MULTIPLIER,
    edge_rgb_color_list,
    rgb_color_list,
    risk_cat_color_dict,
    layout_list,
    bezier_stylesheet,
    round_segment_stylesheet,
    taxi_stylesheet,
    risk_level_color_map,  # Import the new risk_level_color_map
)
from utils import (  # Import needed utility functions
    find_proportional_count_boundaries,
    get_level_from_boundaries,
    get_number_edges_to_show,
)


cyto.load_extra_layouts()


dir_path = os.path.dirname(os.path.realpath(__file__))
new_data_path: GraphDataLibrary = f"{dir_path}/../data/graph/graph_data_library.pkl"


graph_data_library: GraphDataLibrary = pickle.load(open(new_data_path, "rb"))

# Extract unique company names from real_data
companys = sorted(list(graph_data_library.company_graph_datas.keys()))

# Set default company
default_company = companys[0] if companys else None


# --- New function to calculate custom pyramid layout positions ---
def calculate_pyramid_layout(
    selected_node_id, primary_neighbors, secondary_neighbors, subgraph_elements
):
    # Define base positions and spacing
    center_x = 500
    top_y = 100
    primary_y_base = 175
    secondary_y_base = 250
    horizontal_spacing = 150
    vertical_zigzag_offset = 10  # How much to offset vertically for zigzag

    # Create a dictionary to store calculated positions by node ID
    positions = {}

    # Position the selected node at the top center
    positions[selected_node_id] = {"x": center_x, "y": top_y}

    # Position primary neighbors
    num_primary = len(primary_neighbors)
    if num_primary > 0:
        # Calculate starting x position to center the primary nodes
        primary_start_x = center_x - (num_primary - 1) * horizontal_spacing / 2
        for i, node_id in enumerate(primary_neighbors):
            x = primary_start_x + i * horizontal_spacing
            # Apply zigzag vertical offset
            y = primary_y_base + (vertical_zigzag_offset if i % 2 == 1 else 0)
            positions[node_id] = {"x": x, "y": y}

    # Position secondary neighbors
    num_secondary = len(secondary_neighbors)
    if num_secondary > 0:
        # Calculate starting x position to center the secondary nodes
        secondary_start_x = center_x - (num_secondary - 1) * horizontal_spacing / 2
        for i, node_id in enumerate(secondary_neighbors):
            x = secondary_start_x + i * horizontal_spacing
            # Apply zigzag vertical offset (use a different offset or pattern if desired)
            y = secondary_y_base + (
                vertical_zigzag_offset if i % 2 == 0 else 0
            )  # Alternate zigzag pattern
            positions[node_id] = {"x": x, "y": y}

    # Update the position data for each node in the subgraph elements
    updated_elements = []
    for el in subgraph_elements:
        if "source" not in el.get("data", {}):  # It's a node
            node_id = el["data"]["id"]
            if node_id in positions:
                el["position"] = positions[node_id]
        updated_elements.append(el)

    return updated_elements


# --- Helper function to find primary and secondary neighbors ---
def find_neighbors(selected_node_id, all_elements):
    primary_neighbors = set()
    secondary_neighbors = set()
    edges_to_primary = []

    # Find primary neighbors and edges
    for el in all_elements:
        if "source" in el.get("data", {}):  # It's an edge
            source = el["data"]["source"]
            target = el["data"]["target"]
            if source == selected_node_id:
                primary_neighbors.add(target)
                edges_to_primary.append(el)
            elif target == selected_node_id:
                primary_neighbors.add(source)
                edges_to_primary.append(el)

    # Find secondary neighbors
    for el in all_elements:
        if "source" in el.get("data", {}):  # It's an edge
            source = el["data"]["source"]
            target = el["data"]["target"]
            # If the edge connects a primary neighbor to another node
            if (
                source in primary_neighbors
                and target != selected_node_id
                and target not in primary_neighbors
            ):
                secondary_neighbors.add(target)
            elif (
                target in primary_neighbors
                and source != selected_node_id
                and source not in primary_neighbors
            ):
                secondary_neighbors.add(source)

    # Get node elements for selected, primary, and secondary neighbors
    subgraph_nodes = []
    all_neighbor_ids = primary_neighbors.union(secondary_neighbors)
    all_neighbor_ids.add(selected_node_id)  # Include the selected node itself

    for el in all_elements:
        if "source" not in el.get("data", {}):  # It's a node
            if el["data"]["id"] in all_neighbor_ids:
                subgraph_nodes.append(el)

    # Get edges within the subgraph (between selected, primary, and secondary nodes)
    subgraph_edges = []
    for el in all_elements:
        if "source" in el.get("data", {}):  # It's an edge
            source = el["data"]["source"]
            target = el["data"]["target"]
            if source in all_neighbor_ids and target in all_neighbor_ids:
                subgraph_edges.append(el)

    return (
        list(primary_neighbors),
        list(secondary_neighbors),
        subgraph_nodes + subgraph_edges,
    )


# New function to recalculate node raw_size based on a given set of elements (nodes and edges)
def recalculate_node_sizes_based_on_edges(
    elements, node_proportion_list, node_size_list
):
    nodes_in_elements = {}
    for el in elements:
        if "source" not in el.get("data", {}):  # It's a node
            nodes_in_elements[el["data"]["id"]] = el

    # Initialize node raw sizes based on edges within the provided elements
    node_raw_sizes = {node_id: 0.0 for node_id in nodes_in_elements.keys()}

    for el in elements:
        if "source" in el.get("data", {}):  # It's an edge
            src_id = el["data"]["source"]
            tgt_id = el["data"]["target"]
            # Use the raw_weight for node size calculation
            w = el["data"]["raw_weight"]
            if src_id in node_raw_sizes:
                node_raw_sizes[src_id] += w
            if tgt_id in node_raw_sizes:
                node_raw_sizes[tgt_id] += w

    # Get the list of raw sizes for nodes that are actually present in the elements
    current_node_raw_sizes = [
        node_raw_sizes[node_id]
        for node_id in nodes_in_elements.keys()
        if node_raw_sizes[node_id] is not None
    ]

    node_size_counter = Counter()
    if current_node_raw_sizes:
        node_boundaries = find_proportional_count_boundaries(
            current_node_raw_sizes, node_proportion_list
        )

        for node_id, el in nodes_in_elements.items():
            raw_size = node_raw_sizes[node_id]
            level = get_level_from_boundaries(node_boundaries, raw_size)
            display_size = node_size_list[level - 1]
            node_size_counter[level] += 1
            el["data"]["size"] = display_size  # Update node size to display size
    # print(f"{node_size_counter=}")
    return elements


# --- New function to filter elements by weight and recalculate edge properties ---
def filter_elements_by_weight_and_recalculate_edges(
    elements, slider_value, edge_rgb_color_list
):
    node_edge_counter = Counter()
    filtered_elements = []
    old_line_weights = []
    nodes_in_filtered = {}  # Store nodes for easy access

    # First pass: Filter edges and collect raw weights of visible edges
    high_priority_edge_count = 0
    for el in elements:
        if "source" in el.get("data", {}):
            if el["data"]["high_priority"]:
                high_priority_edge_count += 1
                filtered_elements.append(el)
                node_edge_counter["edge"] += 1
                old_line_weights.append(el["data"]["raw_weight"])
    print(f"high_priority_edge_count: {high_priority_edge_count}")
    remaining_slider_value = slider_value - high_priority_edge_count
    for el in elements:
        if "source" in el.get("data", {}):
            # get all edge that have high_priority
            if el["data"]["similarity_rank"] < remaining_slider_value:
                # Append edge for now, will update properties in second pass
                filtered_elements.append(el)
                node_edge_counter["edge"] += 1
                old_line_weights.append(el["data"]["raw_weight"])
        else:
            # Keep all nodes and store them
            filtered_elements.append(el)
            nodes_in_filtered[el["data"]["id"]] = el
            node_edge_counter["node"] += 1

    # Recalculate edge boundaries and update edge properties for visible edges
    if old_line_weights:
        edge_boudaries = find_proportional_count_boundaries(
            old_line_weights, [60, 30, 10]
        )
        for el in filtered_elements:
            if "source" in el.get("data", {}):
                old_line_weight = el["data"]["raw_weight"]
                level = get_level_from_boundaries(edge_boudaries, old_line_weight)
                display_weight = level * EDGE_SIZE_MULTIPLIER
                el["data"]["color"] = edge_rgb_color_list[level - 1]
                el["data"][
                    "weight"
                ] = display_weight  # Update edge weight to display weight

    # --- Removed the node raw_size and display_size recalculation logic from here ---
    # --- New logic to recalculate node raw_size based on filtered edge display weights ---
    # Initialize node raw sizes based on filtered edges
    # node_raw_sizes = {node_id: 0.0 for node_id in nodes_in_filtered.keys()}

    # for el in filtered_elements:
    #     if "source" in el.get("data", {}):
    #         src_id = el["data"]["source"]
    #         tgt_id = el["data"]["target"]
    #         # Use the updated display weight ('weight')
    #         w = el["data"]["raw_weight"]
    #         if src_id in node_raw_sizes:
    #             node_raw_sizes[src_id] += w
    #         if tgt_id in node_raw_sizes:
    #             node_raw_sizes[tgt_id] += w

    # # Get the list of raw sizes in the same order as nodes were added to filtered_elements
    # current_node_raw_sizes = [
    #     node_raw_sizes[el["data"]["id"]]
    #     for el in filtered_elements
    #     if "source" not in el.get("data", {})
    # ]
    # node_size_counter = Counter()
    # # Recalculate node boundaries and update node sizes for visible nodes
    # if current_node_raw_sizes:
    #     # Assuming node_proportion_list and node_size_list are accessible in this scope
    #     # (They are defined globally in the provided context)
    #     node_proportion_list = [65, 30, 5]  # Define or ensure access to these
    #     node_size_list = [1, 50, 120]  # Define or ensure access to these

    #     node_boudaries = find_proportional_count_boundaries(
    #         current_node_raw_sizes, node_proportion_list
    #     )

    #     node_idx = 0
    #     for el in filtered_elements:
    #         if "source" not in el.get("data", {}):
    #             raw_size = current_node_raw_sizes[node_idx]
    #             level = get_level_from_boundaries(node_boudaries, raw_size)
    #             display_size = node_size_list[level - 1]
    #             node_size_counter[level] += 1
    #             el["data"]["size"] = display_size  # Update node size to display size
    #             node_idx += 1
    # --- End of new logic ---
    # print(f"{node_size_counter=}")
    return filtered_elements, node_edge_counter


# New function to process graph data for display
def process_graph_data_for_display(
    graph_data_library: GraphDataLibrary,
    company: str,
):
    nodes = []
    edges = []
    line_weight_list = []
    node_size_multiplier = 10  # This might be superseded by node_size_list
    number_of_scales = 3  # Number of levels for node and edge sizing
    node_proportion_list = [65, 30, 5]  # Proportions for node sizing
    node_size_list = [1, 50, 120]  # Actual sizes for nodes based on level

    company_graph_data: CompanyGraphData = None
    for k, v in graph_data_library.company_graph_datas.items():
        if k == company:
            company_graph_data = v
            break

    if not company_graph_data:
        return [], [], 0, 0

    node_raw_sizes_from_edges = {}
    for node_data_with_embedding in company_graph_data.nodes:
        node_id = node_data_with_embedding.data.id
        node_raw_sizes_from_edges[node_id] = 0.0

    for edge_data in company_graph_data.edges:
        raw_weight = (
            edge_data.cosine_similarity
        )  # raw_weight is -1 to 1 and 1 is the most similar
        line_weight_list.append(raw_weight)

        node_raw_sizes_from_edges[edge_data.source] += raw_weight
        node_raw_sizes_from_edges[edge_data.target] += raw_weight

        edges.append(
            {
                "data": {
                    "source": edge_data.source,
                    "target": edge_data.target,
                    "similarity_rank": edge_data.similarity_rank,
                    "raw_weight": raw_weight,
                    "high_priority": edge_data.high_priority,
                    "do_not_cal_weight": False,  # This might need to be re-evaluated
                    "edge_relation_reason": edge_data.rationale,
                    "source_risk_data": edge_data.risk_a_data.model_dump(),
                    "target_risk_data": edge_data.risk_b_data.model_dump(),
                    "arrow_weight": (
                        "triangle"
                        if edge_data.direction in ["A → B", "Both"]
                        else "none"
                    ),
                }
            }
        )
    if line_weight_list:
        edge_boundaries = find_proportional_count_boundaries(
            line_weight_list, [60, 30, 10]  # Proportions for edge sizing
        )

        for edge_el in edges:
            raw_weight = edge_el["data"]["raw_weight"]
            level = get_level_from_boundaries(edge_boundaries, raw_weight)
            display_weight = level * EDGE_SIZE_MULTIPLIER
            edge_color = edge_rgb_color_list[level - 1]
            edge_el["data"]["weight"] = display_weight
            edge_el["data"]["color"] = edge_color

    current_node_raw_sizes = list(node_raw_sizes_from_edges.values())
    if current_node_raw_sizes:
        node_boundaries = find_proportional_count_boundaries(
            current_node_raw_sizes, node_proportion_list
        )
        for node_data_with_embedding in company_graph_data.nodes:
            node_id = node_data_with_embedding.data.id
            raw_size = node_raw_sizes_from_edges[node_id]
            level = get_level_from_boundaries(node_boundaries, raw_size)
            display_size = node_size_list[level - 1]
            # Use risk_level to determine node fill color
            node_fill_color = risk_level_color_map[
                node_data_with_embedding.data.risk_level
            ]
            # Store risk_cat for outline styling in stylesheet
            risk_category = node_data_with_embedding.data.risk_cat
            # New: Set outline color and width as data properties
            outline_color = risk_cat_color_dict.get(
                risk_category, "#CCCCCC"
            )  # Get color from dict or fallback
            outline_width = 3  # Fixed line width as requested

            nodes.append(
                {
                    "data": {
                        "id": node_id,
                        "label": node_data_with_embedding.data.label,
                        "raw_size": raw_size,
                        "size_level": level,
                        "size": display_size,
                        "color": node_fill_color,  # Node color based on risk_level
                        "risk_level": node_data_with_embedding.data.risk_level,
                        "risk_cat": risk_category,  # Store risk_cat for outline
                        "story": node_data_with_embedding.data.risk_desc_summary or "",
                        "color_outline": outline_color,  # New: Outline color data property
                        "outline_linewidth": outline_width,  # New: Outline width data property
                    },
                    "position": {
                        "x": random.uniform(100, 700),
                        "y": random.uniform(100, 700),
                    },
                }
            )

    total_edges = len(edges)
    total_nodes = len(nodes)
    print(
        f"Generated {total_nodes} nodes and {total_edges} edges for company {company}."
    )
    return nodes + edges, line_weight_list, total_edges, total_nodes


# Capture total_edges in the initial call
elements, line_weights, total_edges, total_nodes = process_graph_data_for_display(
    graph_data_library,
    default_company,
)  # Initial call with empty checklist

app = dash.Dash(__name__, url_base_pathname="/plot_network/")


app.layout = html.Div(
    [
        # --- Add explanation notes ---
        html.Div(
            [
                html.H4("Graph Explanation:"),
                html.P(
                    "Node Size: Represents the influence or centrality of a risk. Larger nodes indicate higher influence."
                ),
                html.P(
                    "Edge Thickness: Represents the similar of risk contents between two risks. Thicker edges indicate higher similarity."
                ),
                html.P(
                    "Arrow on Edge: Indicates a causal relationship. An arrow from Risk A to Risk B means Risk A causes Risk B. For non-arrow edges, it means the risks are not direct dependency but are similar"
                ),
                html.P("Node Colors: Represent risk levels"),
                html.P("Node Highlight/Outline (Risk Category):"),
                html.Ul(
                    [
                        html.Li(
                            f"Operational Risk: {risk_cat_color_dict.get("Operational Risk", "#CCCCCC")}"
                        ),
                        html.Li(
                            f"Strategic Risk: {risk_cat_color_dict.get("Strategic Risk", "#CCCCCC")}"
                        ),
                        html.Li(
                            f"Credit Risk: {risk_cat_color_dict.get("Credit Risk", "#CCCCCC")}"
                        ),
                        html.Li(
                            f"Market Risk: {risk_cat_color_dict.get("Market Risk", "#CCCCCC")}"
                        ),
                        html.Li(
                            f"Liquidity Risk: {risk_cat_color_dict.get("Liquidity Risk", "#CCCCCC")}"
                        ),
                    ]
                ),
            ],
            style={
                "margin": "20px",
                "padding": "15px",
                "border": "1px solid #e0e0e0",
                "border-radius": "8px",
                "background-color": "#f9f9f9",
                "font-size": "0.9em",
                "line-height": "1.6",
            },
        ),
        # --- End of explanation notes ---
        dcc.Dropdown(
            id="company-dropdown",
            options=[{"label": name, "value": name} for name in companys],
            value=default_company,
            clearable=False,
            style={"width": "400px", "margin-bottom": "10px"},
        ),
        dcc.Dropdown(
            id="layout-dropdown",
            options=[{"label": l, "value": l} for l in layout_list],
            value=layout_list[0],
            clearable=False,
            style={"width": "200px", "margin-bottom": "10px"},
        ),
        # Removed the bezier control sliders as per user request
        # html.Div(
        #     [
        #         html.Label("Bezier control-point-step-size"),
        #         dcc.Slider(
        #             id="bezier-step-size-slider",
        #             min=1,
        #             max=50,
        #             step=1,
        #             value=10,
        #             marks={i: str(i) for i in range(1, 51, 5)},
        #             tooltip={"placement": "bottom", "always_visible": False},
        #         ),
        #         html.Label("Bezier control-point-weight"),
        #         dcc.Slider(
        #             id="bezier-weight-slider",
        #             min=0,
        #             max=1,
        #             step=0.01,
        #             value=0.5,
        #             marks={0: "0", 0.5: "0.5", 1: "1"},
        #             tooltip={"placement": "bottom", "always_visible": False},
        #         ),
        #     ],
        #     style={"margin-bottom": "20px"},
        # ),
        # dcc.Checklist(
        #     id="filter-checklist",
        #     options=CHECKLIST_OPTIONS,
        #     value=([CHECKLIST_OPTIONS[0]["value"]] if CHECKLIST_OPTIONS else []),
        #     # value=([]),
        #     inline=True,
        # ),
        html.Div(
            id="checklist-output-container"
        ),  # Keep this Div for now, remove its content in the callback
        html.Hr(),
        # --- New toggle button to hide edges with no arrows ---
        dcc.Checklist(
            id="hide-no-arrow-edges-toggle",
            options=[{"label": "Hide Edges with No Arrows", "value": "hide"}],
            value=[],  # Default to not hidden
            inline=True,
            style={"margin-bottom": "10px"},
        ),
        # --- End of new toggle button ---
        # --- New checkbox to show/hide node outlines ---
        dcc.Checklist(
            id="toggle-node-outline",
            options=[{"label": "Show Node Outlines", "value": "show"}],
            value=["show"],  # Default to checked (show outlines)
            inline=True,
            style={"margin-bottom": "10px"},
        ),
        # --- End of new checkbox ---
        # --- New slider for selecting number of edges ---
        html.Div(
            [
                html.Label("Number of Edges to Display"),
                dcc.Slider(
                    id="num-edges-slider",  # Changed ID
                    min=0,
                    max=total_edges,  # Set max to total edges
                    step=1,
                    value=math.ceil(
                        get_number_edges_to_show(total_nodes)
                    ),  # Default to 10% of total edges, rounded up
                    # Marks will be generated dynamically in the callback
                    # marks={i: str(i) for i in range(0, total_edges + 1, max(1, total_edges // 10))},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        # Removed the old weight-slider
        # dcc.Slider(
        #     id="weight-slider",
        #     min=slider_min,
        #     max=slider_max,
        #     value=initial_slider_value,
        #     step=1,
        #     marks={
        #         i: str(i)
        #         for i in range(math.ceil(slider_min), math.floor(slider_max) + 1, 10)
        #     },
        # ),
        html.Div(id="slider-output-container"),
        html.Div(
            cyto.Cytoscape(
                id="cytospace",
                elements=elements,  # Use initial elements
                layout={"name": "fcose"},  # Set fixed layout
                stylesheet=bezier_stylesheet,  # Use initial stylesheet
                # stylesheet=round_segment_stylesheet,
                style={
                    "width": "100%",
                    "height": "80vh",  # 80% of viewport height
                    "border": "2px solid #ccc",
                    "border-radius": "8px",
                    "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                },
            ),
            style={
                "margin": "20px",
                "padding": "10px",
            },
        ),
        # --- Add a Div to display clicked edge info ---
        html.Div(id="edge-info-output", style={"margin": "20px", "padding": "10px"}),
        # --- End of new Div ---
        # --- Moved dropdown for selecting a node below the main plot ---
        html.Div(
            [
                html.Label("Select a Node"),
                dcc.Dropdown(
                    id="node-dropdown",
                    options=[],  # Options will be populated by callback
                    value=None,
                    clearable=True,
                    placeholder="Select a node...",
                    style={"width": "400px"},
                ),
            ],
            style={
                "margin": "20px",
                "padding": "10px",
            },  # Added margin/padding for spacing
        ),
        # --- New Div for displaying selected node info and secondary plot ---
        html.Div(id="selected-node-info"),  # Text output for primary/secondary nodes
        html.Div(
            cyto.Cytoscape(
                id="subgraph-cytospace",  # New ID for the secondary plot
                elements=[],  # Initially empty
                layout={"name": "cose"},  # Use a standard layout for the subgraph
                stylesheet=bezier_stylesheet,  # Can reuse or define a new stylesheet
                style={
                    "width": "100%",
                    "height": "40vh",  # Smaller height for the subgraph
                    "border": "2px solid #ccc",
                    "border-radius": "8px",
                    "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                    "margin-top": "20px",  # Add space above the subgraph
                },
            ),
            style={
                "margin": "20px",
                "padding": "10px",
            },
        ),
    ]
)


# Callback to update the checklist output
# @app.callback(
#     Output("checklist-output-container", "children"),
#     [Input("filter-checklist", "value")],
# )
# def update_checklist_output(selected_values):
#     if selected_values is None:
#         selected_values = []
#     print(f"Checklist values changed to: {selected_values}")  # Print to console
#     # You can also retrieve the original complex value from CHECKLIST_OPTIONS if needed:
#     return (
#         f"Selected options: {', '.join(selected_values) if selected_values else 'None'}"
#     )


# Callback to update the graph elements and slider output based on company selection
# Remove the two separate callbacks for cytoscape/slider outputs and combine into one:
@app.callback(
    [
        Output("cytospace", "elements"),
        Output("slider-output-container", "children"),
        Output("num-edges-slider", "min"),
        Output("num-edges-slider", "max"),
        Output("num-edges-slider", "value"),
        Output("num-edges-slider", "marks"),
        # Add output for the node dropdown options
        Output("node-dropdown", "options"),
        Output("cytospace", "layout"),
        Output("cytospace", "stylesheet"),
    ],
    [
        Input("company-dropdown", "value"),
        # Removed input for the old weight-slider
        # Input("weight-slider", "value"),
        Input("layout-dropdown", "value"),
        # Removed bezier sliders inputs
        # Input("bezier-step-size-slider", "value"),
        # Input("bezier-weight-slider", "value"),
        # Removed Input("filter-checklist", "value"),
        Input("num-edges-slider", "value"),  # <-- Add input for the new slider
        Input(
            "hide-no-arrow-edges-toggle", "value"
        ),  # <-- Add input for the new toggle
        Input("cytospace", "layout"),  # <-- Add input to get current layout/positions
        Input(
            "toggle-node-outline", "value"
        ),  # <-- Add input for the new outline toggle
    ],
)
def update_graph_and_output(
    company,
    # Removed slider_value input
    # slider_value,
    # Removed layout_name input
    layout_name,
    # Removed bezier_step_size and bezier_weight inputs
    # bezier_step_size,
    # bezier_weight,
    # selected_checklist_values, # Removed this parameter
    num_edges_to_show,
    hide_no_arrow_edges,
    current_cytoscape_layout,  # <-- Add parameter for current layout
    toggle_node_outline_value,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        # No input has triggered the callback yet
        triggered_id = "no_trigger"
    else:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Regenerate elements based on company and checklist selection
    elements, line_weights, total_edges, total_nodes = process_graph_data_for_display(
        graph_data_library,
        company,
        # selected_checklist_values, # Removed this parameter
    )

    # If the company dropdown triggered the callback, reset num_edges_to_show
    if triggered_id == "company-dropdown":
        num_edges_to_show = get_number_edges_to_show(total_nodes)

    if total_edges == 0:
        # No edges, threshold doesn't matter for filtering, but set a high value
        slider_value = float("inf")
        current_num_edges_shown = 0
    else:
        # Ensure num_edges_to_show is within the valid range [0, total_edges]
        # Use the slider's value, or default to 2 * total_nodes if slider value is None/invalid
        if num_edges_to_show is None or not (0 <= num_edges_to_show <= total_edges):
            num_edges_to_show = get_number_edges_to_show(total_nodes)
        # print(f"{num_edges_to_show=}") # Keep for debugging if needed

        if num_edges_to_show == 0:
            # If showing 0 edges, set threshold higher than max weight
            slider_value = max(line_weights) + 1 if line_weights else float("-inf")
        elif num_edges_to_show == total_edges:
            # If showing all edges, set threshold lower than min weight
            slider_value = min(line_weights) - 1 if line_weights else float("-inf")
        else:
            # Sort weights descending and find the weight at the index corresponding to the number of edges
            sorted_weights = sorted(line_weights, reverse=True)
            # The threshold is the weight of the (num_edges_to_show)-th edge (0-indexed)
            # Ensure index is within bounds
            index = min(num_edges_to_show - 1, len(sorted_weights) - 1)
            slider_value = sorted_weights[index]
        current_num_edges_shown = (
            num_edges_to_show  # The number of edges we intend to show
        )

    # Calculate min, max, value, and marks for the new number of edges slider
    num_edges_slider_min = 0
    num_edges_slider_max = total_edges
    # Use the adjusted num_edges_to_show for the slider's value
    num_edges_slider_value = num_edges_to_show

    # Generate marks for the number of edges slider
    if total_edges > 0:
        # Create marks every 10% of the total edges, or at least every 10 edges, or just 0 and max if small
        step = max(1, total_edges // 10)
        marks = {i: str(i) for i in range(0, total_edges + 1, step)}
        # Ensure 0 and total_edges are always included in marks
        marks[0] = "0"
        marks[total_edges] = str(total_edges)
    else:
        marks = {0: "0"}

    # Filter elements based on the calculated slider value and recalculate edge properties
    filtered_elements, node_edge_counter = (
        filter_elements_by_weight_and_recalculate_edges(
            elements, num_edges_to_show, edge_rgb_color_list
        )
    )

    # --- Apply outline visibility based on toggle-node-outline checkbox ---
    show_outline = "show" in toggle_node_outline_value
    for el in filtered_elements:
        if "source" not in el.get("data", {}):  # It's a node
            if not show_outline:
                # el["data"]["color_outline"] = "#000000"  # Set outline color to black
                el["data"]["color_outline"] = "#FFFFFF"  # Set outline color to white
            else:
                # Revert to original risk_cat color if checkbox is checked
                # Ensure the original risk_cat value is available in node data
                node_risk_cat = el["data"].get("risk_cat", None)
                if node_risk_cat:
                    el["data"]["color_outline"] = risk_cat_color_dict.get(
                        node_risk_cat, "#CCCCCC"
                    )
                else:
                    el["data"][
                        "color_outline"
                    ] = "#CCCCCC"  # Fallback if risk_cat is missing
            # Ensure linewidth remains 3px as per the new requirement
            el["data"]["outline_linewidth"] = 3
    # --- End of outline visibility logic ---

    # --- New logic to hide edges with arrow_weight == 0 if toggle is active ---
    # We will modify the elements in place or create a new list with modified styles
    def filter_hide_no_arrow_edges(filtered_elements):
        modified_elements = []
        for el in filtered_elements:
            if "source" in el.get("data", {}):  # It's an edge
                if (
                    "hide" in hide_no_arrow_edges
                    and el["data"].get("arrow_weight") == "none"
                ):
                    pass
                else:
                    # Ensure opacity is 1 for visible edges (or default)
                    el["style"] = {"opacity": 1}
                    modified_elements.append(el)
            else:  # It's a node
                modified_elements.append(el)
        return modified_elements

    filtered_elements = filter_hide_no_arrow_edges(filtered_elements)
    # Update edge count after potentially hiding (by opacity) edges
    # The count should still reflect all edges that passed the weight filter, even if their opacity is 0
    node_edge_counter["edge"] = sum(
        1 for el in filtered_elements if "source" in el.get("data", {})
    )
    # --- End of new logic ---

    # --- New: Recalculate node sizes after filtering edges with no arrows ---
    # Define or ensure access to these lists within this callback if they are not global
    node_proportion_list = [65, 30, 5]  # Proportions for node sizing
    node_size_list = [1, 50, 120]  # Actual sizes for nodes based on level
    filtered_elements = recalculate_node_sizes_based_on_edges(
        filtered_elements, node_proportion_list, node_size_list
    )
    # --- End of new node size recalculation ---

    # --- Generate options for the node dropdown ---
    node_options = []
    # Iterate through filtered_elements to find nodes
    for el in filtered_elements:
        if "source" not in el.get("data", {}):  # Check if it's a node
            node_options.append(
                {
                    "label": el["data"].get(
                        "label", el["data"]["id"]
                    ),  # Use label if available, otherwise id
                    "value": el["data"]["id"],
                }
            )
    print(f"\n\t{node_options=}")
    # Debugging: Print a sample of node data to verify risk_cat
    sample_nodes = [
        el for el in filtered_elements if "source" not in el.get("data", {})
    ][:5]
    print("\n--- Sample Node Data (for risk_cat verification) ---")
    for node in sample_nodes:
        print(json.dumps(node["data"], indent=2))
    print("---------------------------------------------------")

    # --- End of node dropdown options generation ---

    # Update output text to reflect the number of edges shown and the threshold
    output_text = f"Showing {node_edge_counter['edge']} out of {total_edges} edges. Threshold weight: {num_edges_to_show:.2f}. Nodes: {node_edge_counter['node']}"

    # Always use bezier_stylesheet with fixed values for "fcose"
    dynamic_stylesheet = bezier_stylesheet

    return (
        filtered_elements,
        output_text,
        # Removed outputs for the old weight-slider
        # slider_min,
        # slider_max,
        # slider_value,
        # marks,
        # Add outputs for the new number of edges slider
        num_edges_slider_min,
        num_edges_slider_max,
        num_edges_slider_value,
        marks,
        # Return node options
        node_options,
        {"name": layout_name},  # Fixed layout
        dynamic_stylesheet,
    )


# --- New callback to update the subgraph plot and info ---
@app.callback(
    [
        Output("subgraph-cytospace", "elements"),
        Output("subgraph-cytospace", "layout"),
        Output("selected-node-info", "children"),
    ],
    [
        Input("node-dropdown", "value"),
        Input("cytospace", "elements"),  # Get the elements from the main graph
    ],
)
def update_subgraph_and_info(selected_node_id, main_graph_elements):
    if not selected_node_id or not main_graph_elements:
        # Return empty graph and info if no node is selected or main graph is empty
        return [], {"name": "cose"}, ""

    # Find primary and secondary neighbors and the subgraph elements
    primary_neighbors, secondary_neighbors, subgraph_elements = find_neighbors(
        selected_node_id, main_graph_elements
    )

    # --- Create a mapping from node ID to node label from the main graph elements ---
    node_id_to_label = {}
    for el in main_graph_elements:
        if "source" not in el.get("data", {}):  # It's a node
            node_id_to_label[el["data"]["id"]] = el["data"].get(
                "label", el["data"]["id"]
            )  # Use label if available, otherwise id
    # --- End of mapping creation ---

    # Get labels for primary and secondary neighbors
    primary_neighbor_labels = [
        node_id_to_label.get(node_id, node_id) for node_id in primary_neighbors
    ]
    secondary_neighbor_labels = [
        node_id_to_label.get(node_id, node_id) for node_id in secondary_neighbors
    ]
    selected_node_full_data = None
    for el in main_graph_elements:
        if "id" in el.get("data", {}) and el["data"]["id"] == selected_node_id:
            selected_node_full_data = el["data"]
            break
    # Generate text output using labels
    info_text_elements = [
        html.H4(
            f"Selected Node: {node_id_to_label.get(selected_node_id, selected_node_id)} (ID: {selected_node_id})"
        ),
        html.P(
            f"Primary Connections ({len(primary_neighbor_labels)}): {', '.join(primary_neighbor_labels) if primary_neighbor_labels else 'None'}"
        ),
        html.P(
            f"Secondary Connections ({len(secondary_neighbor_labels)}): {', '.join(secondary_neighbor_labels) if secondary_neighbor_labels else 'None'}"
        ),
    ]
    print(f"{selected_node_full_data=}")

    # Add the story if it exists and is not empty
    if selected_node_full_data and "story" in selected_node_full_data:
        node_story = selected_node_full_data["story"]
        if node_story:  # Check if the story is not an empty string
            info_text_elements.append(html.P(f"Story: {node_story}"))

    info_text = html.Div(
        info_text_elements,
        style={
            "margin": "20px",
            "padding": "10px",
            "border": "1px solid #ccc",
            "border-radius": "8px",
        },
    )
    # --- Apply the custom pyramid layout ---
    subgraph_elements_with_positions = calculate_pyramid_layout(
        selected_node_id, primary_neighbors, secondary_neighbors, subgraph_elements
    )
    # --- End of custom layout application ---

    # For a manually positioned layout, we use the 'preset' layout
    subgraph_layout = {"name": "preset"}

    return subgraph_elements_with_positions, subgraph_layout, info_text


# --- New callback to display edge information on click ---
@app.callback(
    Output("edge-info-output", "children"),
    [Input("cytospace", "tapEdge")],
)
def display_edge_info(edge_data):
    if edge_data:
        # Extract relevant data from the clicked edge
        source_id = edge_data["data"].get("source", "N/A")
        target_id = edge_data["data"].get("target", "N/A")
        raw_weight = edge_data["data"].get("raw_weight", "N/A")
        display_weight = edge_data["data"].get("weight", "N/A")
        color = edge_data["data"].get("color", "N/A")
        arrow_weight = edge_data["data"].get("arrow_weight", "N/A")
        do_not_cal_weight = edge_data["data"].get("do_not_cal_weight", "N/A")
        edge_relation_reason = edge_data["data"].get("edge_relation_reason", "N/A")
        source_risk_data = edge_data["data"].get("source_risk_data", {})
        target_risk_data = edge_data["data"].get("target_risk_data", {})
        # You might want to look up the actual node labels here if needed
        # For simplicity, we'll just use the IDs for now
        source_risk_data = json.dumps(source_risk_data, indent=4, ensure_ascii=False)
        target_risk_data = json.dumps(target_risk_data, indent=4, ensure_ascii=False)

        return html.Div(
            [
                html.H5("Clicked Edge Information:"),
                html.P(f"Edge Relation Reason: {edge_relation_reason}"),
                html.P("Source Risk Data:"),
                html.Pre(source_risk_data),
                html.P("Target Risk Data:"),
                html.Pre(target_risk_data),
                html.P(f"Source Node ID: {source_id}"),
                html.P(f"Target Node ID: {target_id}"),
                html.P(
                    f"Raw Weight: {raw_weight:.2f}"
                    if isinstance(raw_weight, (int, float))
                    else f"Raw Weight: {raw_weight}"
                ),
                html.P(
                    f"Display Weight: {display_weight:.2f}"
                    if isinstance(display_weight, (int, float))
                    else f"Display Weight: {display_weight}"
                ),
                html.P(f"Color: {color}"),
                html.P(f"Arrow Weight: {arrow_weight}"),
                html.P(f"Do Not Calculate Weight: {do_not_cal_weight}"),
                # Add more data fields as needed
            ]
        )
    return ""  # Return empty string if no edge is tapped


if __name__ == "__main__":
    # for dev 6060
    # for production 7070
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=6060, help="Port to run the Dash app on"
    )
    args = parser.parse_args()
    app.run(debug=True, port=args.port)


# Add callback to toggle explanation visibility
@app.callback(
    Output("graph-explanation", "style"), [Input("explanation-toggle", "value")]
)
def toggle_explanation(show_explanation):
    if "show" in show_explanation:
        return {
            "display": "block",
            "border": "1px solid #ddd",
            "border-radius": "5px",
            "padding": "10px",
            "margin-bottom": "20px",
            "background-color": "#f9f9f9",
        }
    else:
        return {"display": "none"}
