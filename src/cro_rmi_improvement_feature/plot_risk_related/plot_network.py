import dash
import dash_cytoscape as cyto
from dash import html
from dash import dcc  # Dash Core Components for Slider
from dash.dependencies import Input, Output  # For callbacks
import random
import string
import math
import numpy as np
from .utils import find_equal_count_boundaries


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


def generate_network_from_real_data(data_list):
    """
    data_list: list of dicts, each dict contains:
        - "risk": str
        - "embedding_risk": list or np.array
        - "risk_desc": str
        - "embedding_risk_desc": list or np.array
    """
    number_of_scales = 3
    nodes = []
    edges = []
    line_weight_list = []

    # Create nodes for each risk
    for idx, data in enumerate(data_list):
        nodes.append(
            {
                "data": {
                    "id": f"risk_{idx}",
                    "label": data["risk"],
                    "size": 10,
                    "color": "rgb(54, 162, 235)",
                },
                "position": {
                    "x": random.uniform(100, 700),
                    "y": random.uniform(100, 700),
                },
            }
        )

    # Create edges based on embedding distance (Euclidean)
    # cal the distance between each pair of risks first
    stored_distances = {}
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            emb1 = np.array(data_list[i]["embedding_risk"])
            emb2 = np.array(data_list[j]["embedding_risk"])
            distance = np.linalg.norm(emb1 - emb2)
            line_weight_list.append(distance)
            stored_distances[(i, j)] = distance

    min_dist = min(line_weight_list)
    max_dist = max(line_weight_list)
    # then create edges
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            display_weight = stored_distances[(i, j)]
            # display_weight is seperated into 3 scales based on the min_dist and max_dist
            edges.append(
                {
                    "data": {
                        "source": f"risk_{i}",
                        "target": f"risk_{j}",
                        "weight": math.floor(
                            (stored_distances[(i, j)] - min_dist)
                            / (max_dist - min_dist)
                            * 20
                        ),
                        "raw_weight": stored_distances[(i, j)],
                        "color": "rgb(54, 162, 235)",
                    }
                }
            )

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
file_path = f"{dir_path}/example_data.pkl"

if not os.path.exists(file_path):
    from sentence_transformers import SentenceTransformer

    # Load local embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings for risks and risk_descs
    for item in example_data:
        item["embedding_risk"] = embedder.encode(item["risk"])
        item["embedding_risk_desc"] = embedder.encode(item["risk_desc"])

    # Save the data to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(example_data, f)
else:
    # Load the data from the pickle file
    with open(file_path, "rb") as f:
        example_data = pickle.load(f)

# Use the real data network generator
elements, line_weights = generate_network_from_real_data(example_data)

app = dash.Dash(__name__)

# Generate elements and line weights
# elements, line_weights = generate_az_network()

checkbox_labels = [
    {"label": "risks", "value": ["risks"]},
    {"label": "risks+risk_descs", "value": ["risks", "risk_descs"]},
    {
        "label": "risks+risk_descs+risk_categories",
        "value": ["risks", "risk_descs", "risk_categories"],
    },
    {
        "label": "risks+risk_descs+risk_categories+risk_types",
        "value": ["risks", "risk_descs", "risk_categories", "risk_types"],
    },
]

# Prepare options for the checklist
checklist_options = [
    {"label": item["label"], "value": item["label"]} for item in checkbox_labels
]

# Calculate slider range
if line_weights:  # Ensure list is not empty
    min_weight = min(line_weights)
    max_weight = max(line_weights)
    weight_diff = max_weight - min_weight
    slider_min = min_weight - 0.25 * weight_diff
    slider_max = max_weight + 0.25 * weight_diff
    initial_slider_value = min_weight  # Or any other appropriate initial value
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
            "font-size": "12px",
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

app.layout = html.Div(
    [
        dcc.Checklist(
            id="filter-checklist",
            options=checklist_options,
            value=(
                [checklist_options[0]["value"]] if checklist_options else []
            ),  # Default to the first option selected
            inline=True,
        ),
        html.Div(id="checklist-output-container"),  # To display checklist selection
        html.Hr(),  # Add a horizontal line for separation
        dcc.Slider(
            id="weight-slider",
            min=slider_min,
            max=slider_max,
            value=initial_slider_value,  # Set initial value
            step=1,  # Or choose a step that makes sense for your range
            marks={
                i: str(i)
                for i in range(math.ceil(slider_min), math.floor(slider_max) + 1, 50)
            },  # Optional: marks on the slider
        ),
        html.Div(id="slider-output-container"),  # To display slider value
        cyto.Cytoscape(
            id="cytospace",
            elements=elements,  # Initial elements
            layout={"name": "preset"},
            stylesheet=default_stylesheet,
            style={"width": "800px", "height": "800px"},
        ),
    ]
)


# Callback to update the graph elements and slider output
@app.callback(
    [
        Output("cytospace", "elements"),  # Add Output for cytoscape elements
        Output("slider-output-container", "children"),
    ],
    [Input("weight-slider", "value")],
)
def update_graph_and_output(slider_value):
    filtered_elements = []
    # 'elements' is the global variable holding all original nodes and edges
    for el in elements:
        # Check if the element is an edge by looking for 'source' key in its data
        if "source" in el.get("data", {}):
            if el["data"]["raw_weight"] >= slider_value:
                filtered_elements.append(el)
        else:  # It's a node, always include nodes
            filtered_elements.append(el)

    print(f"Slider value: {slider_value}")  # Print value to console
    # Update the message to reflect the filtering
    output_text = f"Current threshold: {slider_value:.2f}. Showing edges with weight >= {slider_value:.2f}."
    return filtered_elements, output_text


# Callback to update the checklist output
@app.callback(
    Output("checklist-output-container", "children"),
    [Input("filter-checklist", "value")],
)
def update_checklist_output(selected_values):
    if selected_values is None:
        selected_values = []
    print(f"Checklist values changed to: {selected_values}")  # Print to console
    # You can also retrieve the original complex value from checkbox_labels if needed:
    original_selected_data = [
        item for item in checkbox_labels if item["label"] in selected_values
    ]
    print(f"Original data for selected items: {original_selected_data}")
    return (
        f"Selected options: {', '.join(selected_values) if selected_values else 'None'}"
    )


if __name__ == "__main__":
    app.run(debug=True, port=7070)
