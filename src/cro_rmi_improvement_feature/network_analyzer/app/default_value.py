FONT_SIZE = 5
EDGE_SIZE_MULTIPLIER = 2
CHECKLIST_OPTIONS = [
    {"label": "risk_desc_label", "value": "risk_desc"},
    {"label": "rootcause_label", "value": "rootcause"},
    {"label": "process_label", "value": "process"},
]


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

# New: Map risk levels to specific colors (Green, Yellow, Orange, Red)
risk_level_color_map = {
    1: "#7FFF7F",  # Green for Level 1
    2: "#FFFF00",  # Yellow for Level 2
    3: "#FFA500",  # Orange for Level 3
    4: "#FF0000",  # Red for Level 4
}

risk_cat_color_dict = {
    "Operational Risk": "rgb(54, 162, 235)",
    "Strategic Risk": "rgb(255, 206, 86)",
    "Credit Risk": "rgb(75, 192, 192)",
    "Market Risk": "rgb(153, 102, 255)",
    "Liquidity Risk": "rgb(255, 159, 64)",
}
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
            # --- Add arrow properties here ---
            "target-arrow-shape": "data(arrow_weight)",  # Add a triangle arrow at the target end
            "target-arrow-color": "data(color)",  # Make the arrow color match the edge color
            "arrow-scale": "1",  # Adjust arrow size if needed (default is 1)
            # "source-arrow-shape": "circle", # Example for source arrow
            # "source-arrow-color": "blue",
            # --- End of arrow properties ---
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
    # node highlight selectors for risk categories (outline color)
    {
        "selector": 'node[risk_cat = "Operational Risk"]',
        "style": {
            "border-width": "3px",
            "border-color": risk_cat_color_dict.get("Operational Risk", "#CCCCCC"),
            "border-style": "solid",
        },
    },
    {
        "selector": 'node[risk_cat = "Strategic Risk"]',
        "style": {
            "border-width": "3px",
            "border-color": risk_cat_color_dict.get("Strategic Risk", "#CCCCCC"),
            "border-style": "solid",
        },
    },
    {
        "selector": 'node[risk_cat = "Credit Risk"]',
        "style": {
            "border-width": "3px",
            "border-color": risk_cat_color_dict.get("Credit Risk", "#CCCCCC"),
            "border-style": "solid",
        },
    },
    {
        "selector": 'node[risk_cat = "Market Risk"]',
        "style": {
            "border-width": "3px",
            "border-color": risk_cat_color_dict.get("Market Risk", "#CCCCCC"),
            "border-style": "solid",
        },
    },
    {
        "selector": 'node[risk_cat = "Liquidity Risk"]',
        "style": {
            "border-width": "3px",
            "border-color": risk_cat_color_dict.get("Liquidity Risk", "#CCCCCC"),
            "border-style": "solid",
        },
    },
    # Fallback for other risk categories or if not found
    {
        "selector": "node[risk_cat]",  # Selects any node with a risk_cat property
        "style": {
            "border-width": "3px",
            "border-color": "#CCCCCC",  # Default grey border for unmatched categories
            "border-style": "solid",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",  # Use bezier for bundled edges
            # "haystack-radius": "0",
            "opacity": "0.4",
            "line-color": "data(color)",  # Use the color data property for edges
            "width": "mapData(weight, 0, 20, 1, 8)",
            "overlay-padding": "3px",
            "content": "data(weight)",
            "font-size": "0px",  # set to 0px to hide the label
            "text-valign": "center",
            "text-halign": "center",
            # --- Add arrow properties here ---
            "target-arrow-shape": "data(arrow_weight)",  # Add a triangle arrow at the target end
            "target-arrow-color": "data(color)",  # Make the arrow color match the edge color
            "arrow-scale": "1",  # Adjust arrow size if needed (default is 1)
            # "source-arrow-shape": "circle", # Example for source arrow
            # "source-arrow-color": "blue",
            # --- End of arrow properties ---
            "control-point-step-size": "10px",  # Fixed for bezier
            "control-point-weight": "0.5",  # Fixed for bezier
            # "edge-distances": "node-position", # deprecated
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
    {
        "selector": "node[risk_level = 1]",
        "style": {
            "border-width": "3px",
            "border-color": "#7FFF7F",  # dim green
            "border-style": "solid",
        },
    },
    {
        "selector": "node[risk_level = 2]",
        "style": {
            "border-width": "3px",
            "border-color": "#FFFF7F",  # dim yellow
            "border-style": "solid",
        },
    },
    {
        "selector": "node[risk_level = 3]",
        "style": {
            "border-width": "3px",
            "border-color": "#FFB17F",  # dim orange
            "border-style": "solid",
        },
    },
    {
        "selector": "node[risk_level = 4]",
        "style": {
            "border-width": "3px",
            "border-color": "#FF7F7F",  # dim red
            "border-style": "solid",
        },
    },
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
            # --- Add arrow properties here ---
            "target-arrow-shape": "data(arrow_weight)",  # Add a triangle arrow at the target end
            "target-arrow-color": "data(color)",  # Make the arrow color match the edge color
            "arrow-scale": "1",  # Adjust arrow size if needed (default is 1)
            # --- End of arrow properties ---
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
    {
        "selector": "node[risk_level = 1]",
        "style": {
            "border-width": "3px",
            "border-color": "#7FFF7F",  # dim green
            "border-style": "solid",
        },
    },
    {
        "selector": "node[risk_level = 2]",
        "style": {
            "border-width": "3px",
            "border-color": "#FFFF7F",  # dim yellow
            "border-style": "solid",
        },
    },
    {
        "selector": "node[risk_level = 3]",
        "style": {
            "border-width": "3px",
            "border-color": "#FFB17F",  # dim orange
            "border-style": "solid",
        },
    },
    {
        "selector": "node[risk_level = 4]",
        "style": {
            "border-width": "3px",
            "border-color": "#FF7F7F",  # dim red
            "border-style": "solid",
        },
    },
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
            # --- Add arrow properties here ---
            "target-arrow-shape": "data(arrow_weight)",  # Add a triangle arrow at the target end
            "target-arrow-color": "data(color)",  # Make the arrow color match the edge color
            "arrow-scale": "1",  # Adjust arrow size if needed (default is 1)
            # --- End of arrow properties ---
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
