"""
Prediction API endpoint for phishing detection.
"""

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session, save_prediction
from app.ml.features import PhishingFeatureExtractor, extract_heuristic_features
from app.ml.preprocess import preprocess_email
from app.schemas import EmailInput, Explanation, HeuristicFlags, PredictionResponse
from app.utils.explain import explain_prediction
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global model and feature extractor
_model = None
_feature_extractor = None
_model_coefficients = None
_feature_names = []

# Model paths
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model-latest.joblib"
EXTRACTOR_PATH = MODELS_DIR / "feature_extractor-latest.joblib"


def load_model():
    """Load the trained model and feature extractor."""
    global _model, _feature_extractor, _model_coefficients, _feature_names
    
    if _model is not None and _feature_extractor is not None:
        return  # Already loaded
    
    try:
        logger.info("Loading model", path=str(MODEL_PATH))
        _model = joblib.load(MODEL_PATH)
        
        logger.info("Loading feature extractor", path=str(EXTRACTOR_PATH))
        _feature_extractor = joblib.load(EXTRACTOR_PATH)
        
        # Extract model coefficients for linear model
        if hasattr(_model, 'coef_'):
            _model_coefficients = _model.coef_[0]
        
        # Get feature names (TF-IDF + heuristics)
        _feature_names = _feature_extractor.feature_names if hasattr(_feature_extractor, 'feature_names') else []
        
        logger.info("Model loaded successfully")
        
    except FileNotFoundError as e:
        logger.error("Model files not found", error=str(e))
        raise RuntimeError(
            "Model not found. Please train the model first using: python app/models/trainer.py --sample"
        )
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise RuntimeError(f"Failed to load model: {str(e)}")


def get_model():
    """Dependency to ensure model is loaded."""
    if _model is None or _feature_extractor is None:
        load_model()
    return _model, _feature_extractor


@router.post("/predict", response_model=PredictionResponse)
async def predict_email(
    email: EmailInput,
    db: AsyncSession = Depends(get_session),
    model_data: tuple = Depends(get_model)
):
    """
    Predict whether an email is phishing or safe.
    
    Args:
        email: Email input with subject and body
        db: Database session
        model_data: Loaded model and feature extractor
        
    Returns:
        Prediction response with label, score, and explanation
    """
    start_time = time.time()
    model, feature_extractor = model_data
    
    try:
        # Preprocess email
        preprocessed = preprocess_email(email.subject, email.body)
        cleaned_text = preprocessed['cleaned_combined']
        
        # Extract features
        # TF-IDF features
        X_tfidf = feature_extractor.transform([cleaned_text])
        
        # Heuristic features
        heuristic_feats = extract_heuristic_features(email.subject, email.body)
        heuristic_array = np.array([list(heuristic_feats.values())])
        
        # Combine features
        X = np.hstack([X_tfidf, heuristic_array])
        
        # Predict
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0, 1]  # Probability of phishing class
        
        # Determine label
        label = "phishing" if prediction == 1 else "safe"
        
        # Calculate risk score (0-100)
        # Calibrate so that >70 is high risk
        score = float(prediction_proba * 100)
        
        # Generate explanation
        explanation_dict = explain_prediction(
            email.subject,
            email.body,
            prediction_proba,
            feature_vector=X[0] if _model_coefficients is not None else None,
            model_coefficients=_model_coefficients,
            feature_names=_feature_names
        )
        
        # Extract suspicious terms
        suspicious_terms = explanation_dict.get('suspicious_terms', [])
        
        # Build response
        response = PredictionResponse(
            label=label,
            score=score,
            suspicious_terms=suspicious_terms,
            explanation=Explanation(
                token_importances=explanation_dict.get('token_importances', []),
                heuristic_flags=HeuristicFlags(**explanation_dict.get('heuristic_flags', {})),
                risk_factors=explanation_dict.get('risk_factors', []),
                confidence_level=explanation_dict.get('confidence_level', 'Medium')
            )
        )
        
        # Save prediction to database
        try:
            await save_prediction(
                subject=email.subject,
                body=email.body,
                label=label,
                score=score,
                suspicious_terms=suspicious_terms,
                explanation=explanation_dict,
                session=db
            )
        except Exception as db_error:
            logger.warning("Failed to save prediction to database", error=str(db_error))
        
        # Log prediction
        elapsed_time = time.time() - start_time
        logger.info(
            "Prediction completed",
            label=label,
            score=round(score, 2),
            elapsed_ms=round(elapsed_time * 1000, 2)
        )
        
        return response
        
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
