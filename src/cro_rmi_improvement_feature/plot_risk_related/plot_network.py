import dash
import dash_cytoscape as cyto
from dash import html
from dash import dcc  # Dash Core Components for Slider
from dash.dependencies import Input, Output  # For callbacks
import random
import string
import math
import numpy as np
from utils import (
    find_equal_count_boundaries,
    get_level_from_boundaries,
    find_proportional_count_boundaries,
)
import argparse
from collections import Counter

FONT_SIZE = 5
cyto.load_extra_layouts()
EDGE_SIZE_MULTIPLIER = 2
CHECKLIST_OPTIONS = [
    {"label": "risk_desc_label", "value": "risk_desc"},
    {"label": "rootcause_label", "value": "rootcause"},
    {"label": "process_label", "value": "process"},
]


node_highlight_selector_risk_level4 = {
    "selector": "node[risk_level = 4]",
    "style": {
        "border-width": "3px",
        "border-color": "#FF7F7F",  # dim red
        "border-style": "solid",
    },
}
node_highlight_selector_risk_level3 = {
    "selector": "node[risk_level = 3]",
    "style": {
        "border-width": "3px",
        "border-color": "#FFB17F",  # dim orange
        "border-style": "solid",
    },
}
node_highlight_selector_risk_level2 = {
    "selector": "node[risk_level = 2]",
    "style": {
        "border-width": "3px",
        "border-color": "#FFFF7F",  # dim yellow
        "border-style": "solid",
    },
}
node_highlight_selector_risk_level1 = {
    "selector": "node[risk_level = 1]",
    "style": {
        "border-width": "3px",
        "border-color": "#7FFF7F",  # dim green
        "border-style": "solid",
    },
}


rgb_color_list = [
    "rgb(255, 99, 132)",  # Red
    "rgb(54, 162, 235)",  # Blue
    "rgb(255, 206, 86)",  # Yellow
    "rgb(75, 192, 192)",  # Green
    "rgb(153, 102, 255)",  # Purple
    "rgb(255, 159, 64)",  # Orange
    "rgb(201, 203, 207)",  # Grey
    "rgb(255, 99, 71)",  # Tomato
    "rgb(60, 179, 113)",  # MediumSeaGreen
    "rgb(218, 112, 214)",  # Orchid
    "rgb(0, 255, 255)",  # Aqua
    "rgb(240, 230, 140)",  # Khaki
]
edge_rgb_color_list = [
    # very light grey
    "rgb(201, 203, 207)",
    # light grey
    # "rgb(211, 211, 211)",
    # grey
    "rgb(169, 169, 169)",
    # dark grey
    "rgb(128, 128, 128)",
    # very dark grey
    "rgb(80, 80, 80)",
    # black
    "rgb(0, 0, 0)",
]


def generate_az_network():
    # Generate nodes A-Z
    nodes = []
    edges = []
    line_weight_list = []  # Initialize the list to store edge weights

    # Calculate circle layout parameters
    radius = 300
    center_x = 350
    center_y = 350
    node_count = 26

    # Generate nodes
    for i, letter in enumerate(string.ascii_uppercase):
        # Calculate position on a circle for better layout
        angle = (2 * math.pi * i) / node_count
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        # Generate random RGB color
        # random from rgb_color_list
        node_color = random.choice(rgb_color_list)

        nodes.append(
            {
                "data": {
                    "id": letter,
                    "label": letter,
                    "size": random.randint(1, 3) * 30,
                    "color": node_color,  # Add random color here
                },
                "position": {"x": x, "y": y},
            }
        )

    # Generate edges between all nodes
    for i, letter1 in enumerate(string.ascii_uppercase):
        edge_color = random.choice(rgb_color_list)

        for letter2 in string.ascii_uppercase[i + 1 :]:
            current_weight = random.randint(50, 199)
            line_weight_list.append(current_weight)  # Collect edge weight
            edges.append(
                {
                    "data": {
                        "source": letter1,
                        "target": letter2,
                        "weight": math.floor(current_weight / 50) * 5,
                        "raw_weight": current_weight,
                        "color": edge_color,
                    }
                }
            )

    return nodes + edges, line_weight_list  # Return the list of weights


def generate_network_from_real_data(data_list, selected_checklist_values=None):
    """
    data_list: list of dicts, each dict contains:
        - "risk": str
        - "embedding_risk": list or np.array
        - "risk_desc": str
        - "embedding_risk_desc": list or np.array
    """
    node_size_multiplier = 10
    number_of_scales = 3
    node_proportion_list = [65, 30, 5]
    node_size_counter = Counter()
    assert len(node_proportion_list) == number_of_scales
    # node_size_list = [1, 30, 60]

    node_size_list = [1, 50, 120]
    assert len(node_size_list) == number_of_scales
    nodes = []
    edges = []
    line_weight_list = []
    # print checkbox value in this function
    selected_checkbox_value_list = []
    if selected_checklist_values is None:
        selected_checklist_values = []
    for selected_checklist_value in selected_checklist_values:
        selected_checkbox_value_list.append(selected_checklist_value)

    # Create edges based on embedding distance (Euclidean)
    # cal the distance between each pair of risks first
    stored_distances = {}
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            emb1 = np.array(
                data_list[i][
                    tuple(sorted(["risk", "embedding"] + selected_checkbox_value_list))
                ]
            )
            emb2 = np.array(
                data_list[j][
                    tuple(sorted(["risk", "embedding"] + selected_checkbox_value_list))
                ]
            )
            distance = np.linalg.norm(emb1 - emb2)
            distance = 1 / distance * 100
            line_weight_list.append(distance)
            stored_distances[(i, j)] = distance

    min_dist = min(line_weight_list)
    max_dist = max(line_weight_list)
    avg_dist = np.mean(line_weight_list)
    print(f"{avg_dist=}")

    edge_boudaries = find_equal_count_boundaries(line_weight_list, number_of_scales)
    # then create edges
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            raw_weight = stored_distances[(i, j)]
            # display_weight is seperated into 3 scales based on the min_dist and max_dist
            level = get_level_from_boundaries(edge_boudaries, raw_weight)
            display_weight = level * EDGE_SIZE_MULTIPLIER
            edge_color = edge_rgb_color_list[level - 1]
            edges.append(
                {
                    "data": {
                        "source": f"risk_{i}",
                        "target": f"risk_{j}",
                        "weight": display_weight,
                        "raw_weight": raw_weight,
                        "color": edge_color,
                    }
                }
            )

    # Calculate raw_size for each node (sum of all connected edge weights)
    node_raw_sizes = [0.0 for _ in range(len(data_list))]

    for edge in edges:
        src = int(edge["data"]["source"].split("_")[1])
        tgt = int(edge["data"]["target"].split("_")[1])
        w = edge["data"]["raw_weight"]
        node_raw_sizes[src] += w
        node_raw_sizes[tgt] += w

    node_boudaries = find_proportional_count_boundaries(
        node_raw_sizes, node_proportion_list
    )
    # Scale node sizes for display (between 20 and 100)
    min_raw = min(node_raw_sizes)
    max_raw = max(node_raw_sizes)

    def scale_size(raw):
        if max_raw == min_raw:
            return 60  # fallback if all are equal
        return 20 + (raw - min_raw) / (max_raw - min_raw) * (100 - 20)

    all_risk_cat = []
    for data in data_list:
        risk_cat = data["risk_cat"]
        if risk_cat not in all_risk_cat:
            all_risk_cat.append(risk_cat)
    all_risk_cat.sort()
    # Create nodes for each risk
    for idx, data in enumerate(data_list):
        raw_size = node_raw_sizes[idx]
        level = get_level_from_boundaries(node_boudaries, raw_size)
        node_size_counter[level] += 1
        display_size = node_size_list[level - 1]
        risk_cat = data["risk_cat"]
        risk_cat_color = rgb_color_list[all_risk_cat.index(risk_cat)]
        nodes.append(
            {
                "data": {
                    "id": f"risk_{idx}",
                    "label": data["risk"],
                    "raw_size": raw_size,
                    "size": display_size,
                    "color": risk_cat_color,
                    "risk_level": data["risk_level"],
                },
                "position": {
                    "x": random.uniform(100, 700),
                    "y": random.uniform(100, 700),
                },
            }
        )
    print(f"{node_size_counter=}")
    return nodes + edges, line_weight_list


# Example data and embedding generation
example_data = [
    # Operation risks
    {
        "risk": "System outage",
        "risk_desc": "Unexpected downtime of critical systems.",
    },
    {
        "risk": "Process failure",
        "risk_desc": "Breakdown in internal processes affecting operations.",
    },
    {
        "risk": "Supply chain disruption",
        "risk_desc": "Delays or interruptions in the supply chain.",
    },
    # Strategic risks
    {
        "risk": "Mergers gone wrong",
        "risk_desc": "Failed integration after a merger or acquisition.",
    },
    {
        "risk": "Innovation failure",
        "risk_desc": "Inability to keep up with technological advancements.",
    },
    {
        "risk": "Reputation damage",
        "risk_desc": "Negative publicity affecting brand value.",
    },
    # Market risks
    {
        "risk": "Interest rate fluctuation",
        "risk_desc": "Changes in interest rates impacting profitability.",
    },
    {
        "risk": "Currency volatility",
        "risk_desc": "Losses due to changes in foreign exchange rates.",
    },
    {
        "risk": "Market demand shift",
        "risk_desc": "Sudden changes in customer preferences.",
    },
    # Compliance risks
    {
        "risk": "Regulatory breach",
        "risk_desc": "Violation of industry regulations.",
    },
    {
        "risk": "Data privacy violation",
        "risk_desc": "Failure to protect customer data as per laws.",
    },
    {
        "risk": "Reporting error",
        "risk_desc": "Incorrect financial or operational reporting.",
    },
]

# example_data.save to pickle file
import pickle
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
# file_path = f"{dir_path}/example_data.pkl"

# if not os.path.exists(file_path):
#     from sentence_transformers import SentenceTransformer

#     # Load local embedding model
#     embedder = SentenceTransformer("all-MiniLM-L6-v2")

#     # Generate embeddings for risks and risk_descs
#     for item in example_data:
#         item["embedding_risk"] = embedder.encode(item["risk"])
#         item["embedding_risk_desc"] = embedder.encode(item["risk_desc"])

#     # Save the data to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(example_data, f)
# else:
#     # Load the data from the pickle file
#     with open(file_path, "rb") as f:
#         example_data = pickle.load(f)

real_data_path = f"{dir_path}/company_risk_data_with_embedding.pkl"
real_data = pickle.load(open(real_data_path, "rb"))


# Use the real data network generator
# elements, line_weights = generate_network_from_real_data(example_data)

print(f"{real_data[0].keys()=}")
# Extract unique company names from real_data
companys = sorted({item["company"] for item in real_data})

# Set default company
default_company = companys[0] if companys else None


# Initial elements for the default company
def get_elements_for_company(company):
    filtered = [item for item in real_data if item["company"] == company]
    if filtered:
        return generate_network_from_real_data(filtered)
    else:
        return [], []


elements, line_weights = get_elements_for_company(default_company)

app = dash.Dash(__name__, url_base_pathname="/plot_network/")

# Generate elements and line weights
# elements, line_weights = generate_az_network()

# Calculate slider range
if line_weights:  # Ensure list is not empty
    min_weight = min(line_weights)
    max_weight = max(line_weights)
    weight_diff = max_weight - min_weight
    slider_min = min_weight - 0.25 * weight_diff
    slider_max = max_weight + 0.25 * weight_diff
    initial_slider_value = (
        5 * max_weight + min_weight
    ) / 6  # Or any other appropriate initial value
else:  # Default values if no weights (e.g., no edges)
    slider_min = 0
    slider_max = 10
    initial_slider_value = 5


default_stylesheet = [
    {
        "selector": "node",
        "style": {
            "width": "mapData(size, 0, 100, 20, 60)",
            "height": "mapData(size, 0, 100, 20, 60)",
            "content": "data(label)",
            "font-size": f"{FONT_SIZE}px",
            "text-valign": "center",
            "text-halign": "center",
            "background-color": "data(color)",  # Use the color data property
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "haystack",
            "haystack-radius": "0",
            "opacity": "0.4",
            "line-color": "data(color)",  # Use the color data property for edges
            "width": "mapData(weight, 0, 20, 1, 8)",
            "overlay-padding": "3px",
            "content": "data(weight)",
            "font-size": "0px",  # set to 0px to hide the label
            "text-valign": "center",
            "text-halign": "center",
        },
    },
]

# *** Bezier Curve Style with Edge Bundling ***
bezier_stylesheet = [
    {
        "selector": "node",
        "style": {
            "width": "mapData(size, 0, 100, 20, 60)",
            "height": "mapData(size, 0, 100, 20, 60)",
            "content": "data(label)",
            "font-size": f"{FONT_SIZE}px",
            "text-valign": "center",
            "text-halign": "center",
            "background-color": "data(color)",
        },
    },
    # Add this new selector for large nodes
    node_highlight_selector_risk_level1,
    node_highlight_selector_risk_level2,
    node_highlight_selector_risk_level3,
    node_highlight_selector_risk_level4,
    {
        "selector": "edge",
        "style": {
            "curve-style": "unbundled-bezier",
            "control-point-step-size": 10,  # Adjust for bundling strength
            "control-point-weight": 0.5,  # Adjust for bundling shape
            "opacity": 0.6,
            "line-color": "data(color)",
            "width": "mapData(weight, 0, 20, 1, 8)",
            "overlay-padding": "3px",
            "content": "data(weight)",
            "font-size": "0px",
            "text-valign": "center",
            "text-halign": "center",
        },
    },
]


# *** Round-Segment Style ***
round_segment_stylesheet = [
    {
        "selector": "node",
        "style": {
            "width": "mapData(size, 0, 100, 20, 60)",
            "height": "mapData(size, 0, 100, 20, 60)",
            "content": "data(label)",
            "font-size": f"{FONT_SIZE}px",
            "text-valign": "center",
            "text-halign": "center",
            "background-color": "data(color)",
        },
    },
    node_highlight_selector_risk_level1,
    node_highlight_selector_risk_level2,
    node_highlight_selector_risk_level3,
    node_highlight_selector_risk_level4,
    {
        "selector": "edge",
        "style": {
            "curve-style": "segments",
            "segment-distances": "20 80",  # Adjust for segment positioning (percentage along the direct line)
            "segment-weights": "0.3 0.7",  # Adjust for segment positioning (weight towards source/target)
            "line-style": "solid",
            "line-color": "data(color)",
            "width": "mapData(weight, 0, 20, 1, 8)",
            "overlay-padding": "3px",
            "content": "data(weight)",
            "font-size": "0px",
            "text-valign": "center",
            "text-halign": "center",
            "border-width": 1,
            "border-color": "data(color)",
            "border-style": "solid",
            "line-cap": "round",  # Make the ends of segments round
            "line-join": "round",  # Make the corners where segments meet round
        },
    },
]

# *** Taxi Curve Style with Potential for Bundling Effect ***
taxi_stylesheet = [
    {
        "selector": "node",
        "style": {
            "width": "mapData(size, 0, 100, 20, 60)",
            "height": "mapData(size, 0, 100, 20, 60)",
            "content": "data(label)",
            "font-size": f"{FONT_SIZE}px",
            "text-valign": "center",
            "text-halign": "center",
            "background-color": "data(color)",
        },
    },
    node_highlight_selector_risk_level1,
    node_highlight_selector_risk_level2,
    node_highlight_selector_risk_level3,
    node_highlight_selector_risk_level4,
    {
        "selector": "edge",
        "style": {
            "curve-style": "taxi",
            "taxi-direction": "vertical",  # Or 'horizontal' depending on layout
            "taxi-turn": 20,  # Adjust for the number of turns and bundling
            "opacity": 0.6,
            "line-color": "data(color)",
            "width": "mapData(weight, 0, 20, 1, 8)",
            "overlay-padding": "3px",
            "content": "data(weight)",
            "font-size": "0px",
            "text-valign": "center",
            "text-halign": "center",
        },
    },
]
layout_list = [
    "circle",
    "concentric",
    "fcose",
    # "cose",
    "euler",
    "spread",
]

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
            # value=([CHECKLIST_OPTIONS[0]["value"]] if CHECKLIST_OPTIONS else []),
            value=([]),
            inline=True,
        ),
        html.Div(id="checklist-output-container"),
        html.Hr(),
        dcc.Slider(
            id="weight-slider",
            min=slider_min,
            max=slider_max,
            value=initial_slider_value,
            step=1,
            marks={
                i: str(i)
                for i in range(math.ceil(slider_min), math.floor(slider_max) + 1, 10)
            },
        ),
        html.Div(id="slider-output-container"),
        cyto.Cytoscape(
            id="cytospace",
            elements=elements,
            layout={"name": layout_list[0]},
            stylesheet=bezier_stylesheet,
            # stylesheet=round_segment_stylesheet,
            style={"width": "800px", "height": "800px"},
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
        Output("weight-slider", "min"),
        Output("weight-slider", "max"),
        Output("weight-slider", "value"),
        Output("weight-slider", "marks"),
        Output("cytospace", "layout"),
        Output("cytospace", "stylesheet"),
    ],
    [
        Input("company-dropdown", "value"),
        Input("weight-slider", "value"),
        Input("layout-dropdown", "value"),
        Input("bezier-step-size-slider", "value"),
        Input("bezier-weight-slider", "value"),
        Input("filter-checklist", "value"),  # <-- Add this line
    ],
)
def update_graph_and_output(
    company,
    slider_value,
    layout_name,
    bezier_step_size,
    bezier_weight,
    selected_checklist_values,
):
    filtered_data = [item for item in real_data if item["company"] == company]
    elements, line_weights = generate_network_from_real_data(
        filtered_data, selected_checklist_values
    )
    # save_file_path = f"network_{company}_{selected_checklist_values}_{slider_value}.pkl"
    # data = {
    #     "elements": elements,
    #     "line_weights": line_weights,
    # }

    # with open(save_file_path, "wb") as f:
    #     pickle.dump(data, f)
    # Calculate slider range
    if line_weights:
        min_weight = min(line_weights)
        max_weight = max(line_weights)
        weight_diff = max_weight - min_weight
        slider_min = min_weight - 0.25 * weight_diff
        slider_max = max_weight + 0.25 * weight_diff
        if slider_value is None or not (slider_min <= slider_value <= slider_max):
            slider_value = min_weight
        marks = {
            i: str(i)
            for i in range(math.ceil(slider_min), math.floor(slider_max) + 1, 10)
        }
    else:
        slider_min = 0
        slider_max = 10
        slider_value = 5
        marks = {i: str(i) for i in range(0, 11, 2)}
    node_edge_counter = Counter()
    filtered_elements = []
    old_line_weights = []
    for el in elements:
        if "source" in el.get("data", {}):
            if el["data"]["raw_weight"] >= slider_value:
                filtered_elements.append(el)
                node_edge_counter["edge"] += 1
                old_line_weights.append(el["data"]["raw_weight"])

        else:
            filtered_elements.append(el)
            node_edge_counter["node"] += 1
    edge_boudaries = find_proportional_count_boundaries(old_line_weights, [60, 30, 10])
    for el in elements:
        if "source" in el.get("data", {}):
            if el["data"]["raw_weight"] >= slider_value:
                filtered_elements.append(el)
                node_edge_counter["edge"] += 1
                old_line_weight = el["data"]["raw_weight"]
                level = get_level_from_boundaries(edge_boudaries, old_line_weight)
                display_weight = level * EDGE_SIZE_MULTIPLIER
                el["data"]["color"] = edge_rgb_color_list[level - 1]
                el["data"]["weight"] = display_weight

        else:
            filtered_elements.append(el)
            node_edge_counter["node"] += 1
    output_text = f"Current threshold: {slider_value:.2f}. Showing edges with weight >= {slider_value:.2f}.\nNode: {node_edge_counter['node']}, Edge: {node_edge_counter['edge']}"
    # Dynamically update bezier stylesheet with slider values
    dynamic_bezier_stylesheet = [
        {
            "selector": "node",
            "style": bezier_stylesheet[0]["style"],
        },
        # Add this new selector for large nodes
        node_highlight_selector_risk_level1,
        node_highlight_selector_risk_level2,
        node_highlight_selector_risk_level3,
        node_highlight_selector_risk_level4,
        {
            "selector": "edge",
            "style": {
                **bezier_stylesheet[1]["style"],
                "curve-style": "unbundled-bezier",
                "control-point-step-size": bezier_step_size,
                "control-point-weight": bezier_weight,
            },
        },
    ]

    return (
        filtered_elements,
        output_text,
        slider_min,
        slider_max,
        slider_value,
        marks,
        {"name": layout_name},
        dynamic_bezier_stylesheet,
    )


if __name__ == "__main__":
    # for dev 6060
    # for production 7070
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=6060, help="Port to run the Dash app on"
    )
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
