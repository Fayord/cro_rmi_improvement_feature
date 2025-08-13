import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_cytoscape as cyto
import math
import copy

# Initialize the Dash app
app = Dash(__name__)

# Initial graph elements
initial_elements = [
    # Nodes
    {"data": {"id": "one", "label": "Node 1"}, "position": {"x": 50, "y": 50}},
    {"data": {"id": "two", "label": "Node 2"}, "position": {"x": 200, "y": 200}},
    {"data": {"id": "three", "label": "Node 3"}, "position": {"x": 300, "y": 150}},
    {"data": {"id": "four", "label": "Node 4"}, "position": {"x": 150, "y": 300}},
]

# Set a threshold for distance in pixels
EDGE_THRESHOLD = 75

# Define the app layout
app.layout = html.Div(
    [
        html.H1("Dash Cytoscape Drag-and-Drop", style={"textAlign": "center"}),
        html.P(
            f"Drag a node and drop it within {EDGE_THRESHOLD} pixels of another node to create an edge.",
            style={"textAlign": "center"},
        ),
        cyto.Cytoscape(
            id="my-cytoscape",
            layout={"name": "preset", "fit": False},
            style={"width": "100%", "height": "500px"},
            elements=initial_elements,
            # This is critical: we need to enable the user to move nodes.
            # The 'movable: true' is a default, but it's good to be explicit.
            # We also enable `fit: false` so that the layout doesn't reset on updates.
        ),
        # A dcc.Store component to hold the previous state of the elements.
        # This is essential for comparing the old and new positions.
        dcc.Store(id="previous-elements", data=initial_elements),
    ]
)


"""
Callback to handle edge creation after a node is moved and tapped.

Note: Dash Cytoscape does not emit 'elements' updates on drag. We listen to
tapNode, which includes the node's latest position, then update our elements
accordingly and add an edge if within threshold.
"""


@callback(
    Output("my-cytoscape", "elements"),
    Output("previous-elements", "data"),
    Input("my-cytoscape", "tapNode"),
    State("my-cytoscape", "elements"),
    State("previous-elements", "data"),
    prevent_initial_call=True,
)
def update_edges_on_tap(moved_node, elements_state, prev_elements):
    # If no node tap event, return current state
    if not moved_node:
        return elements_state, prev_elements

    # Deep copy elements to mutate safely
    updated_elements = copy.deepcopy(elements_state)

    moved_node_id = moved_node.get("data", {}).get("id")
    moved_pos = moved_node.get("position") or {}

    # Sync the moved node's position into our elements so it doesn't snap back
    for el in updated_elements:
        if "position" in el and el["data"]["id"] == moved_node_id:
            el["position"]["x"] = moved_pos.get("x", el["position"].get("x"))
            el["position"]["y"] = moved_pos.get("y", el["position"].get("y"))
            break

    # Position of the moved node after syncing
    moved_el = next(
        (
            el
            for el in updated_elements
            if el.get("data", {}).get("id") == moved_node_id and "position" in el
        ),
        None,
    )
    if not moved_el:
        return updated_elements, updated_elements

    moved_pos = moved_el["position"]

    # Check proximity with other nodes to create an edge
    for other_el in updated_elements:
        if "position" not in other_el:
            continue
        other_id = other_el["data"]["id"]
        if other_id == moved_node_id:
            continue

        pos1 = moved_pos
        pos2 = other_el["position"]
        distance = math.sqrt(
            (pos1["x"] - pos2["x"]) ** 2 + (pos1["y"] - pos2["y"]) ** 2
        )

        if distance < EDGE_THRESHOLD:
            source_id = moved_node_id
            target_id = other_id
            edge_id = f"edge_{source_id}_{target_id}"

            # Prevent duplicates (both orientations)
            edge_exists = any(
                (
                    e.get("data", {}).get("source") == source_id
                    and e.get("data", {}).get("target") == target_id
                )
                or (
                    e.get("data", {}).get("source") == target_id
                    and e.get("data", {}).get("target") == source_id
                )
                for e in updated_elements
                if "source" in e.get("data", {})
            )

            if not edge_exists:
                updated_elements.append(
                    {"data": {"id": edge_id, "source": source_id, "target": target_id}}
                )

    return updated_elements, updated_elements


# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8080)
