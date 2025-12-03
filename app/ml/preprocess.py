"""
Text preprocessing utilities for phishing email detection.
Handles text cleaning, normalization, and preparation for feature extraction.
"""

import re
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Raw email text (subject or body)
        remove_stopwords: Whether to remove English stopwords
        
    Returns:
        Cleaned and normalized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove email headers (common patterns)
    text = re.sub(r'^(from|to|subject|date|cc|bcc):\s*.+$', '', text, flags=re.MULTILINE)
    
    # Replace URLs with a token (preserve for feature extraction)
    text = re.sub(r'http[s]?://\S+', ' URL_TOKEN ', text)
    text = re.sub(r'www\.\S+', ' URL_TOKEN ', text)
    
    # Replace email addresses with token
    text = re.sub(r'\S+@\S+', ' EMAIL_TOKEN ', text)
    
    # Replace IP addresses with token
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' IP_TOKEN ', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\!\?]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
        text = ' '.join(tokens)
    
    return text.strip()


def extract_raw_features(subject: str, body: str) -> dict:
    """
    Extract raw text features before cleaning.
    These are preserved for heuristic analysis.
    
    Args:
        subject: Email subject line
        body: Email body text
        
    Returns:
        Dictionary of raw features
    """
    combined = f"{subject} {body}"
    
    return {
        'raw_subject': subject,
        'raw_body': body,
        'raw_combined': combined,
        'has_urls': bool(re.search(r'http[s]?://|www\.', combined)),
        'has_ip': bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', combined)),
        'url_count': len(re.findall(r'http[s]?://|www\.', combined)),
        'email_count': len(re.findall(r'\S+@\S+', combined)),
    }


def combine_subject_body(subject: str, body: str, weight_subject: float = 2.0) -> str:
    """
    Combine subject and body with optional subject weighting.
    Subject lines are often more indicative of phishing.
    
    Args:
        subject: Email subject line
        body: Email body text
        weight_subject: How many times to repeat subject (default 2.0 gives it more weight)
        
    Returns:
        Combined text
    """
    if weight_subject > 1.0:
        # Repeat subject to give it more weight in TF-IDF
        subject_repeated = ' '.join([subject] * int(weight_subject))
        return f"{subject_repeated} {body}"
    return f"{subject} {body}"


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    try:
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if len(token) > 1]
    except Exception:
        # Fallback to simple split if NLTK fails
        return text.lower().split()


def preprocess_email(subject: str, body: str, remove_stopwords: bool = False) -> dict:
    """
    Complete preprocessing pipeline for an email.
    
    Args:
        subject: Email subject line
        body: Email body text
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Dictionary containing cleaned text and raw features
    """
    # Extract raw features first
    raw_features = extract_raw_features(subject, body)
    
    # Clean text
    clean_subject = clean_text(subject, remove_stopwords)
    clean_body = clean_text(body, remove_stopwords)
    
    # Combine with subject weighting
    combined = combine_subject_body(clean_subject, clean_body, weight_subject=2.0)
    
    return {
        'cleaned_subject': clean_subject,
        'cleaned_body': clean_body,
        'cleaned_combined': combined,
        'raw_features': raw_features,
    }
