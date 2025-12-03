"""
Main FastAPI application for Phishing Email Detector.
Provides REST API for email phishing detection with metrics and monitoring.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from app.api.predict import router as predict_router
from app.db import get_prediction_stats, get_recent_predictions, get_session, init_db
from app.schemas import (
    HealthResponse,
    ModelMetrics,
    PredictionHistoryItem,
    PredictionStatsResponse,
)
from app.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging(log_level="INFO", json_logs=False)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Phishing Email Detector API",
    description="AI-powered API for detecting phishing emails using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'phishing_detector_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

PREDICTION_COUNT = Counter(
    'phishing_detector_predictions_total',
    'Total number of predictions',
    ['label']
)

REQUEST_LATENCY = Histogram(
    'phishing_detector_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

ERROR_COUNT = Counter(
    'phishing_detector_errors_total',
    'Total number of errors',
    ['endpoint']
)

# Include routers
app.include_router(predict_router, tags=["Predictions"])

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Serve frontend static files
app.mount(
    "/frontend",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Phishing Email Detector API")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
    
    # Check if model exists
    model_path = MODELS_DIR / "model-latest.joblib"
    if not model_path.exists():
        logger.warning(
            "Model not found. Please train the model first using: python app/models/trainer.py --sample"
        )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - redirect to frontend."""
    return """
    <html>
        <head>
            <title>Phishing Email Detector</title>
        </head>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
            <h1>🛡️ AI-Powered Phishing Email Detector</h1>
            <p>Welcome to the Phishing Email Detector API.</p>
            <h2>Quick Links:</h2>
            <ul>
                <li><a href="/frontend/">Web UI (Phishing Email Detector)</a></li>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/metrics">Model Metrics</a></li>
                <li><a href="/prometheus-metrics">Prometheus Metrics</a></li>
            </ul>
            <h2>Example Usage:</h2>
            <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "subject": "Verify your account now",
    "body": "Click here to verify: http://suspicious-link.com"
  }'
            </pre>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_session)):
    """
    Health check endpoint for readiness and liveness probes.
    
    Returns:
        Health status with model and database availability
    """
    model_path = MODELS_DIR / "model-latest.joblib"
    model_loaded = model_path.exists()
    
    # Check database connectivity
    db_connected = True
    try:
        await get_prediction_stats(db)
    except Exception:
        db_connected = False
    
    status = "healthy" if (model_loaded and db_connected) else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        database_connected=db_connected,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/metrics", response_model=ModelMetrics)
async def get_metrics():
    """
    Get model performance metrics.
    
    Returns:
        Model evaluation metrics from the latest training run
    """
    # Find latest metrics file
    metrics_files = list(MODELS_DIR.glob("metrics-*.json"))
    
    if not metrics_files:
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Please train the model first."
        )
    
    # Get most recent metrics file
    latest_metrics_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        return ModelMetrics(**metrics_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load metrics: {str(e)}"
        )


@app.get("/prometheus-metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus-formatted metrics
    """
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/history", response_model=List[PredictionHistoryItem])
async def get_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_session)
):
    """
    Get prediction history.
    
    Args:
        limit: Maximum number of predictions to return (default: 50)
        db: Database session
        
    Returns:
        List of recent predictions
    """
    try:
        predictions = await get_recent_predictions(limit=limit, session=db)
        return predictions
    except Exception as e:
        logger.error("Failed to get prediction history", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@app.get("/stats", response_model=PredictionStatsResponse)
async def get_stats(db: AsyncSession = Depends(get_session)):
    """
    Get aggregate prediction statistics.
    
    Args:
        db: Database session
        
    Returns:
        Prediction statistics
    """
    try:
        stats = await get_prediction_stats(db)
        return PredictionStatsResponse(**stats)
    except Exception as e:
        logger.error("Failed to get statistics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Collect request metrics."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code
        ).inc()
        
        return response
        
    except Exception as e:
        ERROR_COUNT.labels(endpoint=request.url.path).inc()
        raise


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
