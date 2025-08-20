import pytest
import httpx
from dotenv import load_dotenv
import os

# Load environment variables from .env.test
dir_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=f"{dir_path}/../../api/.env.test")


@pytest.fixture(scope="module")
def api_base_url():
    return f"http://172.16.100.50:{os.getenv('PORT')}/rmi_graph_based_recommendation"


@pytest.fixture(scope="module")
def client(api_base_url):
    # Get the authentication token from environment variables
    auth_token = os.getenv("TOKEN_AUTH")
    if not auth_token:
        raise ValueError("TOKEN_AUTH not found in .env.test file")

    headers = {"Authorization": f"Bearer {auth_token}"}
    with httpx.Client(base_url=api_base_url, headers=headers, timeout=60.0) as client:
        yield client
