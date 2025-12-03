"""
Explainability utilities for phishing predictions.
Provides interpretable explanations for model decisions.
"""

from typing import Dict, List, Tuple

import numpy as np

from app.ml.features import extract_heuristic_features, get_suspicious_terms


def get_top_feature_importances(
    feature_vector: np.ndarray,
    model_coefficients: np.ndarray,
    feature_names: List[str],
    top_n: int = 8
) -> List[Tuple[str, float]]:
    """
    Get top features contributing to the prediction.
    
    Args:
        feature_vector: Input feature vector for single sample
        model_coefficients: Model coefficients (for linear models)
        feature_names: Names of all features
        top_n: Number of top features to return
        
    Returns:
        List of (feature_name, importance) tuples
    """
    # Calculate feature contributions (feature * coefficient)
    contributions = feature_vector * model_coefficients
    
    # Get indices of top contributors (absolute values)
    top_indices = np.argsort(np.abs(contributions))[-top_n:][::-1]
    
    # Build result list
    importances = []
    for idx in top_indices:
        if idx < len(feature_names) and contributions[idx] != 0:
            feature_name = feature_names[idx]
            importance = float(contributions[idx])
            importances.append((feature_name, importance))
    
    return importances


def explain_prediction(
    subject: str,
    body: str,
    prediction_proba: float,
    feature_vector: np.ndarray = None,
    model_coefficients: np.ndarray = None,
    feature_names: List[str] = None
) -> Dict:
    """
    Generate comprehensive explanation for a prediction.
    
    Args:
        subject: Email subject line
        body: Email body text
        prediction_proba: Prediction probability (0-1)
        feature_vector: Optional feature vector for the email
        model_coefficients: Optional model coefficients
        feature_names: Optional feature names
        
    Returns:
        Dictionary with explanation details
    """
    explanation = {}
    
    # Get suspicious terms
    suspicious_terms = get_suspicious_terms(subject, body, top_n=8)
    explanation['suspicious_terms'] = suspicious_terms
    
    # Get heuristic features
    heuristic_features = extract_heuristic_features(subject, body)
    
    # Build heuristic flags
    heuristic_flags = {
        'contains_url': bool(heuristic_features.get('has_url', 0)),
        'contains_ip': bool(heuristic_features.get('has_ip_url', 0)),
        'urgent_words': bool(heuristic_features.get('has_urgent_words', 0)),
        'suspicious_keywords': bool(heuristic_features.get('has_suspicious_keywords', 0)),
        'url_count': int(heuristic_features.get('url_count', 0)),
        'suspicious_keyword_count': int(heuristic_features.get('suspicious_keyword_count', 0)),
    }
    explanation['heuristic_flags'] = heuristic_flags
    
    # Get token importances if model info is available
    if feature_vector is not None and model_coefficients is not None and feature_names is not None:
        token_importances = get_top_feature_importances(
            feature_vector,
            model_coefficients,
            feature_names,
            top_n=8
        )
        explanation['token_importances'] = [
            [token, float(importance)] for token, importance in token_importances
        ]
    else:
        # Fallback: use suspicious terms as token importances
        explanation['token_importances'] = [
            [term, 0.1] for term in suspicious_terms[:8]
        ]
    
    # Risk factors summary
    risk_factors = []
    
    if heuristic_flags['urgent_words']:
        risk_factors.append("Contains urgent language")
    
    if heuristic_flags['suspicious_keywords']:
        risk_factors.append(f"Contains {heuristic_flags['suspicious_keyword_count']} suspicious keywords")
    
    if heuristic_flags['contains_url']:
        risk_factors.append(f"Contains {heuristic_flags['url_count']} URL(s)")
    
    if heuristic_flags['contains_ip']:
        risk_factors.append("URL contains IP address")
    
    if prediction_proba > 0.9:
        risk_factors.append("Very high confidence phishing pattern")
    elif prediction_proba > 0.7:
        risk_factors.append("High confidence phishing pattern")
    
    explanation['risk_factors'] = risk_factors
    
    # Confidence level
    if prediction_proba >= 0.9:
        confidence = "Very High"
    elif prediction_proba >= 0.7:
        confidence = "High"
    elif prediction_proba >= 0.5:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    explanation['confidence_level'] = confidence
    
    return explanation


def generate_user_friendly_explanation(label: str, score: float, explanation: Dict) -> str:
    """
    Generate a user-friendly text explanation.
    
    Args:
        label: Prediction label ('phishing' or 'safe')
        score: Risk score (0-100)
        explanation: Explanation dictionary from explain_prediction
        
    Returns:
        Human-readable explanation string
    """
    if label == "phishing":
        text = f"This email appears to be PHISHING (risk score: {score:.1f}/100).\n\n"
        
        if explanation.get('risk_factors'):
            text += "Risk factors identified:\n"
            for factor in explanation['risk_factors']:
                text += f"  • {factor}\n"
        
        if explanation.get('suspicious_terms'):
            text += f"\nSuspicious terms found: {', '.join(explanation['suspicious_terms'][:5])}\n"
        
        text += "\n⚠️ Do not click any links or provide personal information."
    else:
        text = f"This email appears to be SAFE (risk score: {score:.1f}/100).\n\n"
        
        if score > 30:
            text += "Note: Some minor risk indicators were found, but overall the email seems legitimate.\n"
        
        text += "However, always verify sender identity before taking action on sensitive requests."
    
    return text
