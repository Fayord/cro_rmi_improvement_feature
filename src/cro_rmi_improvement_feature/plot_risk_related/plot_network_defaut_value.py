FONT_SIZE = 5
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
    "fcose",
    "circle",
    "concentric",
    # "cose",
    "euler",
    "spread",
]
