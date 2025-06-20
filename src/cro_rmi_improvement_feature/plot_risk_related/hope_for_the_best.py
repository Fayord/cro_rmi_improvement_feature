import dash
import dash_cytoscape as cyto
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        cyto.Cytoscape(
            id="cytoscape",
            layout={"name": "preset"},
            style={"width": "800px", "height": "800px"},
            elements=[
                # Main groups
                {"data": {"id": "group1"}, "position": {"x": 250, "y": 250}},
                {"data": {"id": "group2"}, "position": {"x": 500, "y": 250}},
                {"data": {"id": "group3"}, "position": {"x": 250, "y": 500}},
                # Nested group inside group1
                {
                    "data": {"id": "subgroup1", "parent": "group1"},
                    "position": {"x": 200, "y": 200},
                },
                # Nodes in group1
                {
                    "data": {"id": "a", "parent": "group1"},
                    "position": {"x": 200, "y": 300},
                },
                {
                    "data": {"id": "b", "parent": "group1"},
                    "position": {"x": 300, "y": 300},
                },
                # Nodes in subgroup1
                {
                    "data": {"id": "c", "parent": "subgroup1"},
                    "position": {"x": 150, "y": 150},
                },
                {
                    "data": {"id": "d", "parent": "subgroup1"},
                    "position": {"x": 250, "y": 150},
                },
                # Nodes in group2
                {
                    "data": {"id": "e", "parent": "group2"},
                    "position": {"x": 450, "y": 200},
                },
                {
                    "data": {"id": "f", "parent": "group2"},
                    "position": {"x": 550, "y": 200},
                },
                # Nodes in group3
                {
                    "data": {"id": "g", "parent": "group3"},
                    "position": {"x": 200, "y": 450},
                },
                {
                    "data": {"id": "h", "parent": "group3"},
                    "position": {"x": 300, "y": 450},
                },
                # Edges
                {"data": {"source": "a", "target": "b"}},
                {"data": {"source": "c", "target": "d"}},
                {"data": {"source": "e", "target": "f"}},
                {"data": {"source": "g", "target": "h"}},
                {"data": {"source": "group1", "target": "group2"}},
                {"data": {"source": "group2", "target": "group3"}},
            ],
            stylesheet=[
                {
                    "selector": "node",
                    "style": {
                        "label": "data(id)",
                        "background-color": "#0074D9",
                        "width": 20,
                        "height": 20,
                    },
                },
                {
                    "selector": "$node > node",  # compound node style
                    "style": {
                        "background-color": "#FFDC00",
                        "padding": "20px",
                        "shape": "roundrectangle",
                        "border-color": "#FF851B",
                        "border-width": 2,
                    },
                },
                {
                    "selector": "#subgroup1",  # style for nested group
                    "style": {
                        "background-color": "#2ECC40",
                        "padding": "15px",
                        "shape": "roundrectangle",
                        "border-color": "#3D9970",
                        "border-width": 2,
                    },
                },
            ],
        )
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
