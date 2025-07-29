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
    "Operational Risk": "#36A2EB",  # rgb(54, 162, 235)
    "Strategic Risk": "#FFCE56",  # rgb(255, 206, 86)
    "Credit Risk": "#4BC0C0",  # rgb(75, 192, 192)
    "Market Risk": "#9966FF",  # rgb(153, 102, 255)
    "Liquidity Risk": "#FF9F40",  # rgb(255, 159, 64)
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
            "border-width": "data(outline_linewidth)",  # Use data property for outline width
            "border-color": "data(color_outline)",  # Use data property for outline color
            "border-style": "solid",
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
            "border-width": "data(outline_linewidth)",  # Use data property for outline width
            "border-color": "data(color_outline)",  # Use data property for outline color
            "border-style": "solid",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "unbundled-bezier",  # Use unbundled-bezier for curved edges
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
            "border-width": "data(outline_linewidth)",  # Use data property for outline width
            "border-color": "data(color_outline)",  # Use data property for outline color
            "border-style": "solid",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "straight",  # Use straight for round-segment
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
        },
    },
]


# *** Taxi Style ***
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
            "border-width": "data(outline_linewidth)",  # Use data property for outline width
            "border-color": "data(color_outline)",  # Use data property for outline color
            "border-style": "solid",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "taxi",  # Use taxi for taxi style
            "taxi-direction": "auto",
            "taxi-turn": "50%",
            "taxi-radius": "10px",
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
