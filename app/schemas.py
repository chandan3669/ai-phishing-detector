"""
Pydantic schemas for API request and response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class EmailInput(BaseModel):
    """Input schema for email prediction."""
    
    subject: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Email subject line"
    )
    body: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Email body text"
    )
    
    @validator('subject', 'body')
    def validate_not_empty(cls, v):
        """Ensure fields are not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "subject": "Verify your account now",
                "body": "Click here to verify your account: http://suspicious-link.com"
            }
        }


class HeuristicFlags(BaseModel):
    """Heuristic feature flags."""
    
    contains_url: bool
    contains_ip: bool
    urgent_words: bool
    suspicious_keywords: bool
    url_count: int
    suspicious_keyword_count: int


class Explanation(BaseModel):
    """Explanation for prediction."""
    
    token_importances: List[List] = Field(
        description="List of [token, importance] pairs"
    )
    heuristic_flags: HeuristicFlags
    risk_factors: List[str] = Field(
        description="Human-readable risk factors"
    )
    confidence_level: str = Field(
        description="Confidence level: Low, Medium, High, Very High"
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    
    label: str = Field(
        description="Classification label: 'phishing' or 'safe'"
    )
    score: float = Field(
        ge=0,
        le=100,
        description="Risk score from 0 (safe) to 100 (phishing)"
    )
    suspicious_terms: List[str] = Field(
        description="List of suspicious keywords found in the email"
    )
    explanation: Explanation
    
    class Config:
        schema_extra = {
            "example": {
                "label": "phishing",
                "score": 87.3,
                "suspicious_terms": ["verify", "account", "click here", "urgent"],
                "explanation": {
                    "token_importances": [["verify", 0.45], ["urgent", 0.22]],
                    "heuristic_flags": {
                        "contains_url": True,
                        "contains_ip": False,
                        "urgent_words": True,
                        "suspicious_keywords": True,
                        "url_count": 1,
                        "suspicious_keyword_count": 4
                    },
                    "risk_factors": [
                        "Contains urgent language",
                        "Contains 4 suspicious keywords",
                        "Contains 1 URL(s)"
                    ],
                    "confidence_level": "High"
                }
            }
        }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    
    version: str
    timestamp: str
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_roc_auc: float
    training_samples: int
    test_samples: int


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str


class PredictionHistoryItem(BaseModel):
    """Single prediction history item."""
    
    id: int
    timestamp: str
    subject: str
    body_preview: Optional[str]
    label: str
    score: float
    suspicious_terms: List[str]
    explanation: Dict


class PredictionStatsResponse(BaseModel):
    """Prediction statistics response."""
    
    total_predictions: int
    phishing_count: int
    safe_count: int
    phishing_rate: float
    average_score: float
