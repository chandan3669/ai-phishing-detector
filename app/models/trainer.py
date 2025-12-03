"""
Model training pipeline for phishing email detection.
Handles data loading, preprocessing, feature extraction, model training, and evaluation.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ml.features import PhishingFeatureExtractor
from app.ml.preprocess import preprocess_email

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)


def create_sample_dataset() -> pd.DataFrame:
    """
    Create a sample dataset for quick testing and demonstration.
    
    Returns:
        DataFrame with sample phishing and legitimate emails
    """
    # Sample phishing emails
    phishing_emails = [
        {
            "subject": "URGENT: Your account has been suspended",
            "body": "Dear user, Your account has been suspended due to unusual activity. Click here to verify your identity immediately: http://fake-bank-login.com/verify",
            "label": 1
        },
        {
            "subject": "Verify your PayPal account now",
            "body": "We detected unauthorized access to your PayPal account. Please confirm your identity by clicking this link and entering your password.",
            "label": 1
        },
        {
            "subject": "You've won $1,000,000!",
            "body": "Congratulations! You have been selected to receive a cash prize of $1,000,000. Click here to claim your prize now. Limited time offer!",
            "label": 1
        },
        {
            "subject": "Action Required: Update your billing information",
            "body": "Your payment method has expired. Update your billing information immediately to avoid service interruption. Click here: http://192.168.1.1/update",
            "label": 1
        },
        {
            "subject": "IRS Tax Refund - Act Now",
            "body": "You are eligible for a tax refund of $849.32. Click below to validate your identity and receive your refund. Expires in 24 hours!",
            "label": 1
        },
        {
            "subject": "Security Alert: Unusual Activity Detected",
            "body": "We noticed unusual login attempts on your account. Verify your account now to prevent suspension. Click here immediately.",
            "label": 1
        },
        {
            "subject": "Your Amazon order confirmation",
            "body": "Your order #AZ-8429384 for $599.99 has been confirmed. If you did not make this purchase, click here to cancel and verify your account.",
            "label": 1
        },
        {
            "subject": "FINAL NOTICE: Your account will be closed",
            "body": "This is your final notice. Your account will be permanently closed in 48 hours unless you verify your information. Click here urgently.",
            "label": 1
        },
    ]
    
    # Sample legitimate emails
    legitimate_emails = [
        {
            "subject": "Team meeting tomorrow at 2 PM",
            "body": "Hi everyone, Just a reminder that we have our weekly team meeting tomorrow at 2 PM in conference room B. Please bring your status updates.",
            "label": 0
        },
        {
            "subject": "Project deadline extended",
            "body": "Good news! The deadline for the Q4 project has been extended to next Friday. This should give everyone more time to complete their tasks.",
            "label": 0
        },
        {
            "subject": "Lunch plans for Friday?",
            "body": "Hey, want to grab lunch this Friday? I was thinking we could try that new Italian restaurant downtown. Let me know if you're available!",
            "label": 0
        },
        {
            "subject": "Monthly newsletter - January 2024",
            "body": "Welcome to our monthly newsletter. This month's highlights include new feature releases, team updates, and upcoming events. Read more on our blog.",
            "label": 0
        },
        {
            "subject": "Question about the documentation",
            "body": "Hi, I was reviewing the API documentation and had a question about the authentication flow. Could you clarify how refresh tokens work?",
            "label": 0
        },
        {
            "subject": "Family reunion photos",
            "body": "Hi everyone! I've uploaded all the photos from last weekend's family reunion to the shared album. Feel free to download and share them!",
            "label": 0
        },
        {
            "subject": "Conference schedule for next week",
            "body": "Attached is the finalized schedule for the tech conference next week. Your presentation is on Wednesday at 10 AM. Looking forward to it!",
            "label": 0
        },
        {
            "subject": "Book recommendation",
            "body": "I just finished reading an excellent book on system design that I think you'd enjoy. It has great insights on scalability patterns. Want to borrow it?",
            "label": 0
        },
    ]
    
    all_emails = phishing_emails + legitimate_emails
    df = pd.DataFrame(all_emails)
    
    return df


def load_and_prepare_data(sample_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare training data.
    
    Args:
        sample_mode: If True, use sample dataset instead of loading from files
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if sample_mode:
        print("Using sample dataset...")
        df = create_sample_dataset()
    else:
        # Check for existing processed data
        processed_file = PROCESSED_DATA_DIR / "combined_emails.csv"
        
        if processed_file.exists():
            print(f"Loading processed data from {processed_file}")
            df = pd.read_csv(processed_file)
        else:
            print("No processed data found. Using sample dataset.")
            print("To use real data, place CSV files with 'subject', 'body', 'label' columns in data/raw/")
            df = create_sample_dataset()
            # Save for future use
            df.to_csv(processed_file, index=False)
    
    # Ensure required columns exist
    required_columns = ['subject', 'body', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Handle missing values
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    print(f"Training samples: {len(train_df)} (Phishing: {sum(train_df['label'])}, Safe: {sum(1-train_df['label'])})")
    print(f"Test samples: {len(test_df)} (Phishing: {sum(test_df['label'])}, Safe: {sum(1-test_df['label'])})")
    
    return train_df, test_df


def preprocess_dataset(df: pd.DataFrame) -> Tuple[list, list, list, np.ndarray]:
    """
    Preprocess a dataset.
    
    Args:
        df: DataFrame with 'subject', 'body', 'label' columns
        
    Returns:
        Tuple of (cleaned_texts, raw_subjects, raw_bodies, labels)
    """
    cleaned_texts = []
    raw_subjects = []
    raw_bodies = []
    
    for _, row in df.iterrows():
        preprocessed = preprocess_email(row['subject'], row['body'])
        cleaned_texts.append(preprocessed['cleaned_combined'])
        raw_subjects.append(row['subject'])
        raw_bodies.append(row['body'])
    
    labels = df['label'].values
    
    return cleaned_texts, raw_subjects, raw_bodies, labels


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_version: str = None
) -> Dict:
    """
    Train phishing detection model.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        model_version: Optional version string for model filename
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*50)
    print("Starting Model Training")
    print("="*50)
    
    # Preprocess data
    print("\nPreprocessing training data...")
    train_texts, train_subjects, train_bodies, y_train = preprocess_dataset(train_df)
    
    print("Preprocessing test data...")
    test_texts, test_subjects, test_bodies, y_test = preprocess_dataset(test_df)
    
    # Feature extraction
    print("\nExtracting features...")
    feature_extractor = PhishingFeatureExtractor(
        max_features=3000,
        ngram_range=(1, 3),
        use_char_ngrams=True
    )
    
    # Fit on training data
    feature_extractor.fit(train_texts)
    
    # Transform with heuristics
    X_train = feature_extractor.transform_with_heuristics(
        train_texts, train_subjects, train_bodies
    )
    X_test = feature_extractor.transform_with_heuristics(
        test_texts, test_subjects, test_bodies
    )
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Train model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    
    # Training metrics
    print("\nTraining Set:")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    
    # Test metrics
    print("\nTest Set:")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"ROC AUC:   {test_roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Safe', 'Phishing']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
    
    # Save model and artifacts
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filename = MODELS_DIR / f"model-v{model_version}.joblib"
    extractor_filename = MODELS_DIR / f"feature_extractor-v{model_version}.joblib"
    metrics_filename = MODELS_DIR / f"metrics-v{model_version}.json"
    
    print(f"\nSaving model to {model_filename}")
    joblib.dump(model, model_filename)
    
    print(f"Saving feature extractor to {extractor_filename}")
    joblib.dump(feature_extractor, extractor_filename)
    
    # Save metrics
    metrics = {
        "version": model_version,
        "timestamp": datetime.now().isoformat(),
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "test_roc_auc": float(test_roc_auc),
        "confusion_matrix": cm.tolist(),
    }
    
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_filename}")
    
    # Create symlinks to latest model
    latest_model = MODELS_DIR / "model-latest.joblib"
    latest_extractor = MODELS_DIR / "feature_extractor-latest.joblib"
    
    # Copy files for Windows compatibility (symlinks require admin on Windows)
    import shutil
    shutil.copy(model_filename, latest_model)
    shutil.copy(extractor_filename, latest_extractor)
    
    print(f"\nLatest model links created")
    print("="*50)
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train phishing detection model")
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample dataset for quick training'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Model version string'
    )
    
    args = parser.parse_args()
    
    # Load data
    train_df, test_df = load_and_prepare_data(sample_mode=args.sample)
    
    # Train model
    metrics = train_model(train_df, test_df, model_version=args.version)
    
    # Print summary
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"\nTest Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Test F1 Score: {metrics['test_f1']:.4f}")
    
    if metrics['test_accuracy'] >= 0.88:
        print("\n✓ Model meets the 88% accuracy target!")
    else:
        print(f"\n✗ Model accuracy ({metrics['test_accuracy']:.2%}) is below the 88% target.")
        print("Consider using a larger dataset or tuning hyperparameters.")


if __name__ == "__main__":
    main()
