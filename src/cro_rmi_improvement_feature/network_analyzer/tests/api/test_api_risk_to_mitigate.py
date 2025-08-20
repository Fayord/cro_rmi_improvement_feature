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
    payload = load_payload("test_no_mitigation.json")
    response = client.post("/recommend_risk_to_mitigate", json=payload)

    assert (
        response.status_code == 200
    ), f"Expected status 200, got {response.status_code}. Response: {response.text}"
    response_data = response.json()
    assert (
        "recommendations_with_tags" in response_data
    ), "Response does not contain 'recommendations_with_tags' key"
    assert isinstance(
        response_data["recommendations_with_tags"], list
    ), "'recommendations_with_tags' should be a list"
    assert (
        len(response_data["recommendations_with_tags"]) > 0
    ), "recommendations_with_tags list should not be empty for no mitigation payload"


def test_recommend_risk_to_mitigate_all_mitigation(client: TestClient):
    payload = load_payload("test_all_mitigation.json")
    response = client.post("/recommend_risk_to_mitigate", json=payload)

    assert (
        response.status_code == 200
    ), f"Expected status 200, got {response.status_code}. Response: {response.text}"
    response_data = response.json()
    assert (
        "recommendations_with_tags" in response_data
    ), "Response does not contain 'recommendations_with_tags' key"
    assert isinstance(
        response_data["recommendations_with_tags"], list
    ), "'recommendations_with_tags' should be a list"
    assert (
        len(response_data["recommendations_with_tags"]) == 0
    ), "recommendations_with_tags list should be empty for all mitigation payload"


def test_recommend_risk_to_mitigate_multiple_user_vary_risk_score(client: TestClient):
    payload = load_payload("test_mutiple_user_vary_risk_score.json")
    response = client.post("/recommend_risk_to_mitigate", json=payload)

    assert (
        response.status_code == 200
    ), f"Expected status 200, got {response.status_code}. Response: {response.text}"
    response_data = response.json()
    assert "recommendations_with_tags" in response_data
    assert isinstance(
        response_data["recommendations_with_tags"], list
    ), "'recommendations_with_tags' should be a list"
    assert (
        len(response_data["recommendations_with_tags"]) == 1
    ), "recommendations_with_tags list should not be empty for multiple user vary risk score payload"
    recommendation_with_tags = response_data["recommendations_with_tags"][0]
    # user_ids should > 1
    assert len(recommendation_with_tags["user_ids"]) > 1
    # with tag is_high_risk is True
    assert recommendation_with_tags["is_high_risk"] == True


def test_recommend_risk_to_mitigate_multiple_user_with_some_mitigation(
    client: TestClient,
):
    payload = load_payload("test_mutiple_user_with_some_mitigation.json")
    response = client.post("/recommend_risk_to_mitigate", json=payload)

    assert (
        response.status_code == 200
    ), f"Expected status 200, got {response.status_code}. Response: {response.text}"
    response_data = response.json()
    assert "recommendations_with_tags" in response_data
    assert isinstance(
        response_data["recommendations_with_tags"], list
    ), "'recommendations_with_tags' should be a list"
    assert (
        len(response_data["recommendations_with_tags"]) == 1
    ), "recommendations_with_tags list should not be empty for multiple user with some mitigation payload"
    recommendation_with_tags = response_data["recommendations_with_tags"][0]
    # user_ids should > 1
    assert len(recommendation_with_tags["user_ids"]) == 1
    # and user_ids is ["user_3"]
    assert recommendation_with_tags["user_ids"] == ["user_3"]
    assert recommendation_with_tags["is_high_risk"] == True
