"""
Integration tests for API endpoints.
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Phishing Email Detector" in response.text
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "database_connected" in data
        assert "timestamp" in data


class TestPredictionEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_phishing_email(self):
        """Test prediction on phishing email."""
        payload = {
            "subject": "URGENT: Verify your account now",
            "body": "Click here to verify your account immediately: http://suspicious-link.com/verify"
        }
        
        response = client.post("/predict", json=payload)
        
        # Check response status
        assert response.status_code == 200
        
        # Check response structure
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert "suspicious_terms" in data
        assert "explanation" in data
        
        # Check data types
        assert data["label"] in ["phishing", "safe"]
        assert isinstance(data["score"], (int, float))
        assert 0 <= data["score"] <= 100
        assert isinstance(data["suspicious_terms"], list)
        
        # Check explanation structure
        assert "token_importances" in data["explanation"]
        assert "heuristic_flags" in data["explanation"]
        assert "risk_factors" in data["explanation"]
        assert "confidence_level" in data["explanation"]
    
    def test_predict_safe_email(self):
        """Test prediction on safe email."""
        payload = {
            "subject": "Team meeting tomorrow",
            "body": "Hi everyone, just a reminder about our team meeting tomorrow at 2 PM in conference room B."
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "label" in data
        assert "score" in data
    
    def test_predict_invalid_input_empty_subject(self):
        """Test prediction with empty subject."""
        payload = {
            "subject": "",
            "body": "Some body text"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_input_empty_body(self):
        """Test prediction with empty body."""
        payload = {
            "subject": "Subject",
            "body": ""
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_input_missing_field(self):
        """Test prediction with missing field."""
        payload = {
            "subject": "Only subject"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestMetricsEndpoints:
    """Test metrics and statistics endpoints."""
    
    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "phishing_count" in data
        assert "safe_count" in data
        assert "phishing_rate" in data
        assert "average_score" in data
    
    def test_history_endpoint(self):
        """Test history endpoint."""
        response = client.get("/history?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_prometheus_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/prometheus-metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"


class TestDocumentation:
    """Test API documentation endpoints."""
    
    def test_swagger_docs(self):
        """Test Swagger documentation."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_docs(self):
        """Test ReDoc documentation."""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
