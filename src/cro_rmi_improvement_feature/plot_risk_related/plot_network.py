import dash
import dash_cytoscape as cyto
from dash import html
import random
import string
import math

def generate_az_network():
    # Generate nodes A-Z
    nodes = []
    edges = []
    
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
        
        nodes.append({
            'data': {
                'id': letter,
                'label': letter,
                'size': random.randint(10, 100)
            },
            'position': {'x': x, 'y': y}
        })
    
    # Generate edges between all nodes
    for i, letter1 in enumerate(string.ascii_uppercase):
        for letter2 in string.ascii_uppercase[i+1:]:
            edges.append({
                'data': {
                    'source': letter1,
                    'target': letter2,
                    'weight': random.randint(1, 20)
                }
            })
    
    return nodes + edges

app = dash.Dash(__name__)

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
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "haystack",
            "haystack-radius": "0",
            "opacity": "0.4",
            "line-color": "#bbb",
            "width": "mapData(weight, 0, 20, 1, 8)",
            "overlay-padding": "3px",
            "content": "data(weight)",
            "font-size": "8px",
            "text-valign": "center",
            "text-halign": "center",
        },
    },
]

app.layout = html.Div(
    [
        cyto.Cytoscape(
            id="cytospace",
            elements=generate_az_network(),
            layout={"name": "preset"},
            stylesheet=default_stylesheet,
            style={'width': '800px', 'height': '800px'}
        )
    ]
)

if __name__ == "__main__":
    app.run(debug=True, port=7070)
