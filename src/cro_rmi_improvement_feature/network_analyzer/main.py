"""
Main entry point for the Network Analyzer application.
"""

# Standard library imports

# Third-party imports

# Local imports


def main():
    """
    Runs the Dash application.
    """
    from network_analyzer.app.main import app

    app.run_server(debug=True)


if __name__ == "__main__":
    main()
