"""
Feature engineering for phishing detection.
Combines TF-IDF features with heuristic-based features.
"""

import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Suspicious keywords commonly found in phishing emails
SUSPICIOUS_KEYWORDS = [
    'verify', 'confirm', 'urgent', 'suspended', 'locked', 'update', 'account',
    'click here', 'click below', 'click link', 'login', 'password', 'security',
    'banking', 'paypal', 'ebay', 'amazon', 'billing', 'invoice', 'payment',
    'expire', 'expiration', 'immediately', 'act now', 'limited time', 'offer',
    'congratulations', 'winner', 'prize', 'claim', 'refund', 'tax', 'irs',
    'validate', 'reactivate', 'compromised', 'unauthorized', 'unusual activity',
    'fraud', 'alert', 'notification', 'action required', 'verify identity',
]

# Urgent/pressure words
URGENT_WORDS = [
    'urgent', 'immediately', 'act now', 'hurry', 'expire', 'expiring',
    'deadline', 'final notice', 'last chance', 'limited time', 'today only',
]


def extract_heuristic_features(subject: str, body: str) -> Dict[str, float]:
    """
    Extract heuristic features based on phishing patterns.
    
    Args:
        subject: Email subject line (raw)
        body: Email body text (raw)
        
    Returns:
        Dictionary of heuristic features
    """
    combined = f"{subject} {body}".lower()
    
    features = {}
    
    # URL-based features
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, combined)
    features['url_count'] = len(urls)
    features['has_url'] = 1.0 if urls else 0.0
    
    # Check for IP addresses in URLs (suspicious)
    features['has_ip_url'] = 1.0 if re.search(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', combined) else 0.0
    
    # Email address count
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    features['email_count'] = len(re.findall(email_pattern, combined))
    
    # Suspicious keyword count
    suspicious_count = sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in combined)
    features['suspicious_keyword_count'] = suspicious_count
    features['has_suspicious_keywords'] = 1.0 if suspicious_count > 0 else 0.0
    
    # Urgent word count
    urgent_count = sum(1 for word in URGENT_WORDS if word in combined)
    features['urgent_word_count'] = urgent_count
    features['has_urgent_words'] = 1.0 if urgent_count > 0 else 0.0
    
    # Length features
    features['subject_length'] = len(subject)
    features['body_length'] = len(body)
    features['total_length'] = len(combined)
    
    # Character-based features
    features['exclamation_count'] = combined.count('!')
    features['question_count'] = combined.count('?')
    features['uppercase_ratio'] = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
    
    # Special character density
    special_chars = sum(1 for c in combined if not c.isalnum() and not c.isspace())
    features['special_char_density'] = special_chars / max(len(combined), 1)
    
    # Number density
    numbers = sum(1 for c in combined if c.isdigit())
    features['number_density'] = numbers / max(len(combined), 1)
    
    # Check for mismatched display name/email (requires parsing, simplified here)
    features['has_reply_to'] = 1.0 if 'reply-to:' in combined.lower() else 0.0
    
    # Check for common phishing phrases
    phishing_phrases = [
        'verify your account', 'confirm your identity', 'update your information',
        'suspend your account', 'unusual activity', 'click here', 'click below',
        'act immediately', 'expire soon', 'limited time offer'
    ]
    features['phishing_phrase_count'] = sum(1 for phrase in phishing_phrases if phrase in combined)
    
    # HTML content indicators
    features['has_html'] = 1.0 if bool(re.search(r'<[^>]+>', combined)) else 0.0
    
    return features


def get_suspicious_terms(subject: str, body: str, top_n: int = 8) -> List[str]:
    """
    Extract suspicious terms from email text.
    
    Args:
        subject: Email subject line
        body: Email body text
        top_n: Number of top suspicious terms to return
        
    Returns:
        List of suspicious terms found
    """
    combined = f"{subject} {body}".lower()
    found_terms = []
    
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in combined:
            found_terms.append(keyword)
            if len(found_terms) >= top_n:
                break
    
    return found_terms


class PhishingFeatureExtractor:
    """
    Combined feature extractor using TF-IDF and heuristics.
    """
    
    def __init__(
        self,
        max_features: int = 3000,
        ngram_range: Tuple[int, int] = (1, 3),
        use_char_ngrams: bool = True,
    ):
        """
        Initialize feature extractor.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams for word features
            use_char_ngrams: Whether to include character n-grams
        """
        # Word-level TF-IDF
        self.word_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            token_pattern=r'\b\w+\b',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )
        
        # Character-level TF-IDF (captures obfuscation patterns)
        self.char_vectorizer = None
        if use_char_ngrams:
            self.char_vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(3, 5),
                analyzer='char',
                min_df=2,
                max_df=0.95,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
            )
        
        self.use_char_ngrams = use_char_ngrams
        self.feature_names = []
    
    def fit(self, texts: List[str], y=None):
        """
        Fit the feature extractors on training data.
        
        Args:
            texts: List of cleaned email texts
            y: Optional labels (not used for unsupervised TF-IDF)
            
        Returns:
            self
        """
        # Fit word-level TF-IDF
        self.word_vectorizer.fit(texts)
        
        # Fit character-level TF-IDF
        if self.use_char_ngrams:
            self.char_vectorizer.fit(texts)
        
        # Store feature names
        self.feature_names = self._get_feature_names()
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors (TF-IDF only).
        
        Args:
            texts: List of cleaned email texts
            
        Returns:
            Feature matrix
        """
        # Word TF-IDF features
        word_features = self.word_vectorizer.transform(texts).toarray()
        
        # Character TF-IDF features
        if self.use_char_ngrams:
            char_features = self.char_vectorizer.transform(texts).toarray()
            return np.hstack([word_features, char_features])
        
        return word_features
    
    def transform_with_heuristics(
        self,
        texts: List[str],
        subjects: List[str],
        bodies: List[str]
    ) -> np.ndarray:
        """
        Transform texts with both TF-IDF and heuristic features.
        
        Args:
            texts: List of cleaned combined texts
            subjects: List of raw subject lines
            bodies: List of raw body texts
            
        Returns:
            Combined feature matrix
        """
        # TF-IDF features
        tfidf_features = self.transform(texts)
        
        # Heuristic features
        heuristic_features = []
        for subject, body in zip(subjects, bodies):
            heuristic_features.append(list(extract_heuristic_features(subject, body).values()))
        
        heuristic_features = np.array(heuristic_features)
        
        # Combine all features
        return np.hstack([tfidf_features, heuristic_features])
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of cleaned email texts
            
        Returns:
            Feature matrix
        """
        return self.fit(texts).transform(texts)
    
    def _get_feature_names(self) -> List[str]:
        """Get all feature names."""
        names = list(self.word_vectorizer.get_feature_names_out())
        if self.use_char_ngrams:
            names.extend([f"char_{name}" for name in self.char_vectorizer.get_feature_names_out()])
        return names
