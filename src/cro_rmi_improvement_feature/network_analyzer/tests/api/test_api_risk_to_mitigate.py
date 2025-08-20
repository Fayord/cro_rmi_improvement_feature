import json
import pytest


@pytest.mark.parametrize(
    "payload_file",
    [
        "test_1_no_mitigation.json",
        "test_2_all_mitigation.json",
    ],
)
def test_recommend_risk_to_mitigate(client, payload_file):
    # Construct the full path to the payload file
    payload_path = (
        f"src/cro_rmi_improvement_feature/network_analyzer/tests/api/{payload_file}"
    )

    # Load the JSON payload from the file
    with open(payload_path, "r") as f:
        payload = json.load(f)

    # Send the POST request to the API endpoint
    response = client.post("/recommend_risk_to_mitigate", json=payload)

    # Assert the response status code
    assert (
        response.status_code == 200
    ), f"Expected status 200, got {response.status_code}. Response: {response.text}"

    # Assert that the response contains the 'recommendations' key
    response_data = response.json()
    assert (
        "recommendations" in response_data
    ), "Response does not contain 'recommendations' key"
    assert isinstance(
        response_data["recommendations"], list
    ), "'recommendations' should be a list"
