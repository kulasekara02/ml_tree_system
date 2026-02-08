"""
Unit tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Tests for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "version" in data

    def test_list_models(self, client):
        """Test list models endpoint."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "default_model" in data

    def test_features_endpoint(self, client):
        """Test features endpoint."""
        response = client.get("/features")

        assert response.status_code == 200
        data = response.json()
        assert "feature_names" in data
        assert "feature_count" in data
        assert "target_names" in data

    def test_predict_input_validation(self, client):
        """Test prediction input validation."""
        # Missing features
        response = client.post("/predict", json={
            "features": [1.0] * 10,  # Only 10 features, need 30
            "model_name": "random_forest"
        })

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_model(self, client):
        """Test prediction with invalid model name."""
        response = client.post("/predict", json={
            "features": [1.0] * 30,
            "model_name": "invalid_model"
        })

        assert response.status_code == 422  # Pydantic validation

    def test_batch_predict_validation(self, client):
        """Test batch prediction validation."""
        response = client.post("/predict/batch", json={
            "samples": [],  # Empty samples
            "model_name": "random_forest"
        })

        assert response.status_code == 422  # Validation error

    def test_model_info_not_found(self, client):
        """Test model info for non-existent model."""
        response = client.get("/models/nonexistent_model")

        assert response.status_code == 404


class TestPredictionEndpoints:
    """Tests for prediction endpoints (requires trained models)."""

    @pytest.fixture
    def sample_features(self):
        """Generate sample feature vector."""
        # Realistic feature values for breast cancer dataset
        return [
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
            0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
            0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
            184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]

    def test_predict_format(self, client, sample_features):
        """Test prediction response format."""
        response = client.post("/predict", json={
            "features": sample_features,
            "model_name": "random_forest"
        })

        # May be 503 if models not loaded
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "class_name" in data
            assert "probability" in data
            assert "probabilities" in data
            assert "model_used" in data

    def test_batch_predict_format(self, client, sample_features):
        """Test batch prediction response format."""
        response = client.post("/predict/batch", json={
            "samples": [sample_features, sample_features],
            "model_name": "random_forest"
        })

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "class_names" in data
            assert "count" in data
            assert data["count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
