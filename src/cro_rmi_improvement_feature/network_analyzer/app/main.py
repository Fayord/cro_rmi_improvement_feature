"""
Main Dash application entry point.
"""

# Standard library imports

# Third-party imports
import dash

# Local imports
from .layout import serve_layout
from .callbacks import register_callbacks


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = serve_layout

register_callbacks(app)


if __name__ == "__main__":
    app.run_server(debug=True)
