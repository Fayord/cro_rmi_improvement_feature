import dash
import dash_cytoscape as cyto
from dash import html

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
            "width": "mapData(weight, 0, 30, 1, 8)",
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
            elements=[
                {
                    "data": {"id": "one", "label": "Node 1", "size": 10},
                    "position": {"x": 50, "y": 50},
                },
                {
                    "data": {"id": "three", "label": "Node 3", "size": 40},
                    "position": {"x": 100, "y": 200},
                },
                {
                    "data": {"id": "two", "label": "Node 2", "size": 120},
                    "position": {"x": 200, "y": 200},
                },
                {
                    "data": {
                        "source": "one",
                        "target": "two",
                        "label": "Node 1 to 2",
                        "weight": 50,
                    }
                },
                {
                    "data": {
                        "source": "one",
                        "target": "three",
                        "label": "Node 1 to 2",
                        "weight": 4,
                    }
                },
            ],
            layout={"name": "preset"},
            stylesheet=default_stylesheet,
        )
    ]
)

if __name__ == "__main__":
    app.run(debug=True, port=7070)
