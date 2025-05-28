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


# New function to generate dynamic stylesheet
def generate_dynamic_stylesheet(
    bezier_stylesheet,
    bezier_step_size,
    bezier_weight,
    node_highlight_selector_risk_level1,
    node_highlight_selector_risk_level2,
    node_highlight_selector_risk_level3,
    node_highlight_selector_risk_level4,
):
    dynamic_bezier_stylesheet = [
        {
            "selector": "node",
            "style": bezier_stylesheet[0]["style"],
        },
        node_highlight_selector_risk_level1,
        node_highlight_selector_risk_level2,
        node_highlight_selector_risk_level3,
        node_highlight_selector_risk_level4,
        {
            "selector": "edge",
            "style": {
                "curve-style": "unbundled-bezier",
                "control-point-step-size": bezier_step_size,
                "control-point-weight": bezier_weight,
                "opacity": 0.6,
                "line-color": "data(color)",
                "width": "mapData(weight, 0, 20, 1, 8)",
                "overlay-padding": "3px",
                "content": "data(weight)",
                "font-size": "0px",
                "text-valign": "center",
                "text-halign": "center",
                # --- Add arrow properties here ---
                "target-arrow-shape": "data(arrow_weight)",
                "target-arrow-color": "data(color)",  # Make the arrow color match the edge color
                "arrow-scale": "1",  # Adjust arrow size if needed (default is 1)
                # "source-arrow-shape": "circle", # Example for source arrow
                # "source-arrow-color": "blue",
                # --- End of arrow properties ---
            },
        },
    ]
    return dynamic_bezier_stylesheet


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
