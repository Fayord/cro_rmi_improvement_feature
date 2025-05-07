import dash
import dash_cytoscape as cyto
from dash import html
from dash import dcc  # Dash Core Components for Slider
from dash.dependencies import Input, Output  # For callbacks
import random
import string
import math


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
            current_weight = random.randint(1, 3) * 5
            line_weight_list.append(current_weight)  # Collect edge weight
            edges.append(
                {
                    "data": {
                        "source": letter1,
                        "target": letter2,
                        "weight": current_weight,
                        "color": edge_color,
                    }
                }
            )

    return nodes + edges, line_weight_list  # Return the list of weights


app = dash.Dash(__name__)

# Generate elements and line weights
elements, line_weights = generate_az_network()

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
        dcc.Slider(
            id="weight-slider",
            min=slider_min,
            max=slider_max,
            value=initial_slider_value,  # Set initial value
            step=0.1,  # Or choose a step that makes sense for your range
            marks={
                i: str(i)
                for i in range(math.ceil(slider_min), math.floor(slider_max) + 1)
            },  # Optional: marks on the slider
        ),
        html.Div(id="slider-output-container"),  # To display slider value
        cyto.Cytoscape(
            id="cytospace",
            elements=elements,  # Use the generated elements
            layout={"name": "preset"},
            stylesheet=default_stylesheet,
            style={"width": "800px", "height": "800px"},
        ),
    ]
)


# Callback to update the slider output and print value
@app.callback(
    Output("slider-output-container", "children"), [Input("weight-slider", "value")]
)
def update_output(value):
    print(f"Slider value: {value}")  # Print value to console
    return f"Current slider value: {value}"


if __name__ == "__main__":
    app.run(debug=True, port=7070)
