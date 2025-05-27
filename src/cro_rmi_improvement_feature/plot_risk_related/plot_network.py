import dash
import dash_cytoscape as cyto
from dash import html
from dash import dcc  # Dash Core Components for Slider
from dash.dependencies import Input, Output  # For callbacks
import random
import pickle
import os
import math
import numpy as np
from utils import (
    get_level_from_boundaries,
    find_proportional_count_boundaries,
    get_elements_for_company,
    generate_dynamic_stylesheet,
    calculate_pyramid_layout,
    find_neighbors,
)
import argparse
from collections import Counter

from plot_network_defaut_value import (
    EDGE_SIZE_MULTIPLIER,
    CHECKLIST_OPTIONS,
    edge_rgb_color_list,
    node_highlight_selector_risk_level4,
    node_highlight_selector_risk_level3,
    node_highlight_selector_risk_level2,
    node_highlight_selector_risk_level1,
    layout_list,
    bezier_stylesheet,
    round_segment_stylesheet,
    taxi_stylesheet,
)

cyto.load_extra_layouts()


dir_path = os.path.dirname(os.path.realpath(__file__))

real_data_path = f"{dir_path}/250520-company_risk_data_with_embedding.pkl"
real_data = pickle.load(open(real_data_path, "rb"))


print(f"{real_data[0].keys()=}")
# Extract unique company names from real_data
companys = sorted({item["company"] for item in real_data})

# Set default company
default_company = companys[0] if companys else None


# New function to filter elements by weight and recalculate edge properties
def filter_elements_by_weight_and_recalculate_edges(
    elements, slider_value, edge_rgb_color_list
):
    node_edge_counter = Counter()
    filtered_elements = []
    old_line_weights = []
    nodes_in_filtered = {}  # Store nodes for easy access

    # First pass: Filter edges and collect raw weights of visible edges
    for el in elements:
        if "source" in el.get("data", {}):
            if el["data"]["raw_weight"] >= slider_value:
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

    # --- New logic to recalculate node raw_size based on filtered edge display weights ---
    # Initialize node raw sizes based on filtered edges
    node_raw_sizes = {node_id: 0.0 for node_id in nodes_in_filtered.keys()}

    for el in filtered_elements:
        if "source" in el.get("data", {}):
            src_id = el["data"]["source"]
            tgt_id = el["data"]["target"]
            # Use the updated display weight ('weight')
            w = el["data"]["weight"]
            if src_id in node_raw_sizes:
                node_raw_sizes[src_id] += w
            if tgt_id in node_raw_sizes:
                node_raw_sizes[tgt_id] += w

    # Get the list of raw sizes in the same order as nodes were added to filtered_elements
    current_node_raw_sizes = [
        node_raw_sizes[el["data"]["id"]]
        for el in filtered_elements
        if "source" not in el.get("data", {})
    ]
    node_size_counter = Counter()
    # Recalculate node boundaries and update node sizes for visible nodes
    if current_node_raw_sizes:
        # Assuming node_proportion_list and node_size_list are accessible in this scope
        # (They are defined globally in the provided context)
        node_proportion_list = [65, 30, 5]  # Define or ensure access to these
        node_size_list = [1, 50, 120]  # Define or ensure access to these
        number_of_scales = len(node_proportion_list)  # Or len(node_size_list)

        node_boudaries = find_proportional_count_boundaries(
            current_node_raw_sizes, node_proportion_list
        )

        node_idx = 0
        for el in filtered_elements:
            if "source" not in el.get("data", {}):
                raw_size = current_node_raw_sizes[node_idx]
                level = get_level_from_boundaries(node_boudaries, raw_size)
                display_size = node_size_list[level - 1]
                node_size_counter[level] += 1
                el["data"]["size"] = display_size  # Update node size to display size
                node_idx += 1
    # --- End of new logic ---
    print(f"{node_size_counter=}")
    return filtered_elements, node_edge_counter


# Capture total_edges in the initial call
elements, line_weights, total_edges = get_elements_for_company(
    real_data, default_company, ["risk_desc"]
)  # Initial call with empty checklist


app = dash.Dash(__name__, url_base_pathname="/plot_network/")


app.layout = html.Div(
    [
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
        # --- Add sliders for bezier controls ---
        html.Div(
            [
                html.Label("Bezier control-point-step-size"),
                dcc.Slider(
                    id="bezier-step-size-slider",
                    min=1,
                    max=50,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(1, 51, 5)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Label("Bezier control-point-weight"),
                dcc.Slider(
                    id="bezier-weight-slider",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.5,
                    marks={0: "0", 0.5: "0.5", 1: "1"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        dcc.Checklist(
            id="filter-checklist",
            options=CHECKLIST_OPTIONS,
            value=([CHECKLIST_OPTIONS[0]["value"]] if CHECKLIST_OPTIONS else []),
            # value=([]),
            inline=True,
        ),
        html.Div(id="checklist-output-container"),
        html.Hr(),
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
                        total_edges * 0.10
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
                layout={"name": layout_list[0]},
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
@app.callback(
    Output("checklist-output-container", "children"),
    [Input("filter-checklist", "value")],
)
def update_checklist_output(selected_values):
    if selected_values is None:
        selected_values = []
    print(f"Checklist values changed to: {selected_values}")  # Print to console
    # You can also retrieve the original complex value from CHECKLIST_OPTIONS if needed:
    return (
        f"Selected options: {', '.join(selected_values) if selected_values else 'None'}"
    )


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
        Input("bezier-step-size-slider", "value"),
        Input("bezier-weight-slider", "value"),
        Input("filter-checklist", "value"),
        Input("num-edges-slider", "value"),  # <-- Add input for the new slider
    ],
)
def update_graph_and_output(
    company,
    # Removed slider_value input
    # slider_value,
    layout_name,
    bezier_step_size,
    bezier_weight,
    selected_checklist_values,
    num_edges_to_show,  # <-- Add input parameter for the new slider value
):
    # Regenerate elements based on company and checklist selection
    elements, line_weights, total_edges = get_elements_for_company(
        real_data, company, selected_checklist_values
    )

    # --- New logic to calculate slider_value (weight threshold) based on num_edges_to_show ---
    if total_edges == 0:
        # No edges, threshold doesn't matter for filtering, but set a high value
        slider_value = float("inf")
        current_num_edges_shown = 0
    else:
        # Ensure num_edges_to_show is within the valid range [0, total_edges]
        num_edges_to_show = max(0, min(total_edges, num_edges_to_show))

        if num_edges_to_show == 0:
            # If showing 0 edges, set threshold higher than max weight
            slider_value = max(line_weights) + 1 if line_weights else float("inf")
        elif num_edges_to_show == total_edges:
            # If showing all edges, set threshold lower than min weight
            slider_value = min(line_weights) - 1 if line_weights else float("-inf")
        else:
            # Sort weights descending and find the weight at the index corresponding to the number of edges
            sorted_weights = sorted(line_weights, reverse=True)
            # The threshold is the weight of the (num_edges_to_show)-th edge (0-indexed)
            slider_value = sorted_weights[num_edges_to_show - 1]
        current_num_edges_shown = (
            num_edges_to_show  # The number of edges we intend to show
        )

    # Calculate min, max, value, and marks for the new number of edges slider
    num_edges_slider_min = 0
    num_edges_slider_max = total_edges
    # If the current num_edges_to_show is outside the new range, reset it to the max (show all)
    if num_edges_to_show is None or not (
        num_edges_slider_min <= num_edges_to_show <= num_edges_slider_max
    ):
        num_edges_slider_value = total_edges
    else:
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
            elements, slider_value, edge_rgb_color_list
        )
    )

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
    # --- End of node dropdown options generation ---

    # Update output text to reflect the number of edges shown and the threshold
    output_text = f"Showing {node_edge_counter['edge']} out of {total_edges} edges. Threshold weight: {slider_value:.2f}. Nodes: {node_edge_counter['node']}"

    # Dynamically update stylesheet based on layout and bezier slider values
    if (
        layout_name == "unbundled-bezier"
    ):  # Assuming bezier stylesheet is used with this layout
        dynamic_stylesheet = generate_dynamic_stylesheet(
            bezier_stylesheet,
            bezier_step_size,
            bezier_weight,
            node_highlight_selector_risk_level1,
            node_highlight_selector_risk_level2,
            node_highlight_selector_risk_level3,
            node_highlight_selector_risk_level4,
        )
    elif (
        layout_name == "segments"
    ):  # Assuming round_segment stylesheet is used with this layout
        dynamic_stylesheet = round_segment_stylesheet
    elif layout_name == "taxi":  # Assuming taxi stylesheet is used with this layout
        dynamic_stylesheet = taxi_stylesheet
    else:  # Default to bezier stylesheet for other layouts or if layout_name is None
        dynamic_stylesheet = generate_dynamic_stylesheet(
            bezier_stylesheet,
            bezier_step_size,
            bezier_weight,
            node_highlight_selector_risk_level1,
            node_highlight_selector_risk_level2,
            node_highlight_selector_risk_level3,
            node_highlight_selector_risk_level4,
        )

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
        {"name": layout_name},
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
    for el in elements:
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

        # You might want to look up the actual node labels here if needed
        # For simplicity, we'll just use the IDs for now

        return html.Div(
            [
                html.H5("Clicked Edge Information:"),
                html.P(f"Edge Relation Reason: {edge_relation_reason}"),
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
