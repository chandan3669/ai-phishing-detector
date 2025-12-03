"""
Database models and setup for prediction history.
Uses SQLite with async support via aiosqlite.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database setup
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "phishing_detector.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"
SYNC_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create base class
Base = declarative_base()


class PredictionHistory(Base):
    """Store prediction history for auditing and analysis."""
    
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    subject = Column(String(500), nullable=False)
    body_preview = Column(String(1000), nullable=True)  # Truncated for privacy
    label = Column(String(20), nullable=False)  # 'phishing' or 'safe'
    score = Column(Float, nullable=False)  # Risk score 0-100
    suspicious_terms = Column(Text, nullable=True)  # JSON array
    explanation = Column(Text, nullable=True)  # JSON object
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'subject': self.subject,
            'body_preview': self.body_preview,
            'label': self.label,
            'score': self.score,
            'suspicious_terms': json.loads(self.suspicious_terms) if self.suspicious_terms else [],
            'explanation': json.loads(self.explanation) if self.explanation else {},
        }


# Async engine and session
async_engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        yield session


async def save_prediction(
    subject: str,
    body: str,
    label: str,
    score: float,
    suspicious_terms: List[str],
    explanation: dict,
    session: AsyncSession
):
    """
    Save a prediction to the database.
    
    Args:
        subject: Email subject
        body: Email body
        label: Prediction label
        score: Risk score
        suspicious_terms: List of suspicious terms
        explanation: Explanation dictionary
        session: Database session
    """
    # Truncate body for privacy (store only preview)
    body_preview = body[:1000] if body else ""
    
    prediction = PredictionHistory(
        subject=subject,
        body_preview=body_preview,
        label=label,
        score=score,
        suspicious_terms=json.dumps(suspicious_terms),
        explanation=json.dumps(explanation)
    )
    
    session.add(prediction)
    await session.commit()
    
    return prediction


async def get_recent_predictions(limit: int = 50, session: AsyncSession = None) -> List[dict]:
    """
    Get recent predictions from the database.
    
    Args:
        limit: Maximum number of predictions to return
        session: Database session
        
    Returns:
        List of prediction dictionaries
    """
    from sqlalchemy import select
    
    stmt = select(PredictionHistory).order_by(
        PredictionHistory.timestamp.desc()
    ).limit(limit)
    
    result = await session.execute(stmt)
    predictions = result.scalars().all()
    
    return [p.to_dict() for p in predictions]


async def get_prediction_stats(session: AsyncSession) -> dict:
    """
    Get aggregate statistics on predictions.
    
    Args:
        session: Database session
        
    Returns:
        Statistics dictionary
    """
    from sqlalchemy import func, select
    
    # Total count
    total_stmt = select(func.count(PredictionHistory.id))
    total_result = await session.execute(total_stmt)
    total_count = total_result.scalar()
    
    # Phishing count
    phishing_stmt = select(func.count(PredictionHistory.id)).where(
        PredictionHistory.label == 'phishing'
    )
    phishing_result = await session.execute(phishing_stmt)
    phishing_count = phishing_result.scalar()
    
    # Safe count
    safe_count = total_count - phishing_count
    
    # Average score
    avg_stmt = select(func.avg(PredictionHistory.score))
    avg_result = await session.execute(avg_stmt)
    avg_score = avg_result.scalar() or 0.0
    
    return {
        'total_predictions': total_count,
        'phishing_count': phishing_count,
        'safe_count': safe_count,
        'phishing_rate': phishing_count / total_count if total_count > 0 else 0,
        'average_score': float(avg_score),
    }


def create_tables():
    """Create database tables synchronously (for initial setup)."""
    engine = create_engine(SYNC_DATABASE_URL)
    Base.metadata.create_all(engine)
    print(f"Database created at {DB_PATH}")


if __name__ == "__main__":
    create_tables()
