import pytest
import httpx


@pytest.fixture(scope="module")
def api_base_url():
    # Replace with your actual API base URL, e.g., "http://localhost:8000"
    return "http://localhost:8000/rmi_graph_based_recommendation"


@pytest.fixture(scope="module")
def client(api_base_url):
    with httpx.Client(base_url=api_base_url) as client:
        yield client
