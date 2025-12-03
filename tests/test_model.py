"""
Unit tests for model and ML components.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.preprocess import clean_text, extract_raw_features, preprocess_email
from app.ml.features import extract_heuristic_features, get_suspicious_terms, SUSPICIOUS_KEYWORDS


class TestPreprocessing:
    """Test text preprocessing functions."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "URGENT: Click Here NOW!"
        cleaned = clean_text(text)
        # The function removes special characters including ! and :
        assert "urgent" in cleaned.lower()
        assert "click" in cleaned.lower()
        assert "here" in cleaned.lower()
        assert "now" in cleaned.lower()
    
    def test_clean_text_urls(self):
        """Test URL replacement."""
        text = "Visit http://example.com or www.test.com"
        cleaned = clean_text(text)
        # URLs are replaced with URL_TOKEN (case may vary after lowercasing)
        assert "url_token" in cleaned.lower()
        assert "http" not in cleaned.lower()
    
    def test_clean_text_emails(self):
        """Test email replacement."""
        text = "Contact us at support@example.com"
        cleaned = clean_text(text)
        # Emails are replaced with EMAIL_TOKEN
        assert "email_token" in cleaned.lower()
        assert "@" not in cleaned
    
    def test_clean_text_ip_addresses(self):
        """Test IP address replacement."""
        text = "Go to 192.168.1.1 for more"
        cleaned = clean_text(text)
        # IP addresses are replaced with IP_TOKEN
        assert "ip_token" in cleaned.lower()
        assert "192.168.1.1" not in cleaned
    
    def test_extract_raw_features(self):
        """Test raw feature extraction."""
        subject = "Urgent: Verify your account"
        body = "Click http://suspicious.com to verify"
        features = extract_raw_features(subject, body)
        
        assert features['has_urls'] is True
        assert features['url_count'] >= 1
        assert 'raw_subject' in features
        assert 'raw_body' in features
    
    def test_preprocess_email(self):
        """Test complete email preprocessing."""
        subject = "URGENT ACTION"
        body = "Click here http://test.com"
        result = preprocess_email(subject, body)
        
        assert 'cleaned_subject' in result
        assert 'cleaned_body' in result
        assert 'cleaned_combined' in result
        assert 'raw_features' in result


class TestFeatures:
    """Test feature extraction."""
    
    def test_heuristic_features_phishing(self):
        """Test heuristic features on phishing email."""
        subject = "URGENT: Verify your account NOW!"
        body = "Click here immediately: http://192.168.1.1/verify"
        features = extract_heuristic_features(subject, body)
        
        assert features['has_url'] > 0
        assert features['has_ip_url'] > 0
        assert features['has_urgent_words'] > 0
        assert features['has_suspicious_keywords'] > 0
        assert features['url_count'] >= 1
    
    def test_heuristic_features_safe(self):
        """Test heuristic features on safe email."""
        subject = "Team meeting tomorrow"
        body = "Hi everyone, just a reminder about our meeting at 2 PM."
        features = extract_heuristic_features(subject, body)
        
        assert features['has_url'] == 0
        assert features['has_ip_url'] == 0
        assert features['suspicious_keyword_count'] == 0
    
    def test_get_suspicious_terms(self):
        """Test suspicious term extraction."""
        subject = "Verify your account"
        body = "Click here to confirm your password"
        terms = get_suspicious_terms(subject, body, top_n=5)
        
        assert len(terms) > 0
        assert any(term in SUSPICIOUS_KEYWORDS for term in terms)
    
    def test_get_suspicious_terms_safe_email(self):
        """Test minimal suspicious terms in safe email."""
        subject = "Team meeting notes"
        body = "Here are the notes from our weekly team meeting."
        terms = get_suspicious_terms(subject, body, top_n=5)
        
        # Safe emails should have few or no suspicious terms
        assert len(terms) <= 1


class TestModelTraining:
    """Test model training components."""
    
    def test_sample_dataset_creation(self):
        """Test sample dataset creation."""
        from app.models.trainer import create_sample_dataset
        
        df = create_sample_dataset()
        
        assert len(df) > 0
        assert 'subject' in df.columns
        assert 'body' in df.columns
        assert 'label' in df.columns
        assert df['label'].isin([0, 1]).all()
        
        # Check class balance
        phishing_count = (df['label'] == 1).sum()
        safe_count = (df['label'] == 0).sum()
        assert phishing_count > 0
        assert safe_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
