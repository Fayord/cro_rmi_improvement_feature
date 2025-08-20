import json
import pytest
import os
from fastapi.testclient import TestClient


def load_payload(file_name: str):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    payload_path = f"{dir_path}/{file_name}"
    with open(payload_path, "r") as f:
        return json.load(f)


def test_recommend_risk_to_mitigate_no_mitigation(client: TestClient):
    payload = load_payload("test_1_no_mitigation.json")
    response = client.post("/recommend_risk_to_mitigate", json=payload)

    assert (
        response.status_code == 200
    ), f"Expected status 200, got {response.status_code}. Response: {response.text}"
    response_data = response.json()
    assert (
        "recommendations" in response_data
    ), "Response does not contain 'recommendations' key"
    assert isinstance(
        response_data["recommendations"], list
    ), "'recommendations' should be a list"
    assert (
        len(response_data["recommendations"]) > 0
    ), "Recommendations list should not be empty for no mitigation payload"


def test_recommend_risk_to_mitigate_all_mitigation(client: TestClient):
    payload = load_payload("test_2_all_mitigation.json")
    response = client.post("/recommend_risk_to_mitigate", json=payload)

    assert (
        response.status_code == 200
    ), f"Expected status 200, got {response.status_code}. Response: {response.text}"
    response_data = response.json()
    assert (
        "recommendations" in response_data
    ), "Response does not contain 'recommendations' key"
    assert isinstance(
        response_data["recommendations"], list
    ), "'recommendations' should be a list"
    assert (
        len(response_data["recommendations"]) == 0
    ), "Recommendations list should be empty for all mitigation payload"
