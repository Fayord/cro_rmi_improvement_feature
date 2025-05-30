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
from graph_layout import (
    generate_dynamic_stylesheet,
    calculate_pyramid_layout,
    find_neighbors,
)
import json
import argparse
from collections import Counter

from plot_network_defaut_value import (
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
from data_processing import (
    filter_elements_by_weight_and_recalculate_edges,
    get_elements_for_company,
)

cyto.load_extra_layouts()


dir_path = os.path.dirname(os.path.realpath(__file__))

# real_data_path = f"{dir_path}/250520-company_risk_data_with_embedding.pkl"
real_data_path = f"{dir_path}/result/merge-company_risk_data_with_embedding.pkl"
real_data = pickle.load(open(real_data_path, "rb"))
edge_relationship_path = real_data_path.replace(".pkl", "-edge_relationship.json")

print(f"{real_data[0].keys()=}")
# Extract unique company names from real_data
companys = sorted({item["company"] for item in real_data})

# Set default company
default_company = companys[0] if companys else None


# Capture total_edges in the initial call
elements, line_weights, total_edges = get_elements_for_company(
    real_data,
    default_company,
    edge_relationship_path,
    ["risk_desc"],
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
                html.P("Node Colors: Represent risk categories"),
                html.P("Node Highlight/Outline (4 Levels of Risk):"),
                html.Ul(
                    [
                        html.Li("Level 1: Green  - Description for Level 1 risks."),
                        html.Li("Level 2: Yellow - Description for Level 2 risks."),
                        html.Li("Level 3: Orange - Description for Level 3 risks."),
                        html.Li("Level 4: Red    - Description for Level 4 risks."),
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
        # --- New toggle button to hide edges with no arrows ---
        dcc.Checklist(
            id="hide-no-arrow-edges-toggle",
            options=[{"label": "Hide Edges with No Arrows", "value": "hide"}],
            value=[],  # Default to not hidden
            inline=True,
            style={"margin-bottom": "10px"},
        ),
        # --- End of new toggle button ---
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
        Input(
            "hide-no-arrow-edges-toggle", "value"
        ),  # <-- Add input for the new toggle
        Input("cytospace", "layout"),  # <-- Add input to get current layout/positions
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
    num_edges_to_show,
    hide_no_arrow_edges,
    current_cytoscape_layout,  # <-- Add parameter for current layout
):
    # Regenerate elements based on company and checklist selection
    elements, line_weights, total_edges = get_elements_for_company(
        real_data,
        company,
        edge_relationship_path,
        selected_checklist_values,
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

    # --- New logic to hide edges with arrow_weight == 0 if toggle is active ---
    # We will modify the elements in place or create a new list with modified styles
    modified_elements = []
    for el in filtered_elements:
        if "source" in el.get("data", {}):  # It's an edge
            if (
                "hide" in hide_no_arrow_edges
                and el["data"].get("arrow_weight") == "none"
            ):
                # Set opacity to 0 for hidden edges
                el["style"] = {"opacity": 0}
            else:
                # Ensure opacity is 1 for visible edges (or default)
                el["style"] = {"opacity": 1}
            modified_elements.append(el)
        else:  # It's a node
            modified_elements.append(el)
    filtered_elements = modified_elements

    # Update edge count after potentially hiding (by opacity) edges
    # The count should still reflect all edges that passed the weight filter, even if their opacity is 0
    node_edge_counter["edge"] = sum(
        1 for el in filtered_elements if "source" in el.get("data", {})
    )
    # --- End of new logic ---

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
