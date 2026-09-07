"""
FastAPI endpoint tests — using TestClient (no real server needed).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app
from backend.services.evidence_provider import WebResult
from tests.conftest import MockEvidenceProvider


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def patched_pipeline(mock_provider: MockEvidenceProvider):
    """Patch TavilyEvidenceProvider with mock for API tests."""
    mock_provider.set_response("delhi", [
        WebResult(
            title="Delhi Weather",
            url="https://example.com/delhi",
            content="Heavy rain in Delhi reported by IMD.",
            score=0.9,
        ),
    ])
    with patch(
        "backend.services.verification_pipeline.TavilyEvidenceProvider",
        return_value=mock_provider,
    ):
        yield


class TestHealthEndpoint:

    def test_health(self, client: TestClient):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestVerifyEndpoint:

    def test_verify_with_claim(self, client: TestClient, patched_pipeline):
        response = client.post(
            "/api/v1/claims/verify",
            json={
                "segments": [
                    {
                        "start": 4.0,
                        "end": 8.0,
                        "text": "It is raining heavily in Delhi right now.",
                    }
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Should have transcript
        assert len(data["transcript"]) == 1

        # Should have at least one claim
        assert len(data["claims"]) >= 1

        # Claim should have required fields
        claim = data["claims"][0]
        assert "claim_id" in claim
        assert "start" in claim
        assert "end" in claim
        assert "text" in claim
        assert "claim_type" in claim
        assert "verification" in claim
        assert "risk_contribution" in claim

        # Verification should have status
        assert claim["verification"]["status"] in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]

        # Risk should be present
        assert "risk" in data
        assert "claim_risk_score" in data["risk"]
        assert "risk_level" in data["risk"]

    def test_verify_no_claims(self, client: TestClient, patched_pipeline):
        """Non-factual input should return empty claims."""
        response = client.post(
            "/api/v1/claims/verify",
            json={
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "Hello sir."},
                    {"start": 2.0, "end": 4.0, "text": "How are you?"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["claims"]) == 0

    def test_verify_empty_segments(self, client: TestClient):
        """Empty segments should return 422."""
        response = client.post(
            "/api/v1/claims/verify",
            json={"segments": []},
        )
        assert response.status_code == 422

    def test_verify_missing_segments(self, client: TestClient):
        """Missing segments field should return 422."""
        response = client.post(
            "/api/v1/claims/verify",
            json={},
        )
        assert response.status_code == 422

    def test_verify_text_endpoint(self, client: TestClient, patched_pipeline):
        """Raw text endpoint should auto-segment and verify claims."""
        response = client.post(
            "/api/v1/claims/verify-text",
            json={
                "text": "Hello sir. It is raining heavily in Delhi right now. Thank you.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["transcript"]) >= 1
        assert len(data["claims"]) >= 1
        assert data["claims"][0]["claim_type"] == "WEATHER"

    def test_frontend_served(self, client: TestClient):
        """Frontend index.html should be served at root /."""
        response = client.get("/")
        assert response.status_code == 200
        assert "VaniRakshak" in response.text

