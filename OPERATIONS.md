# Operations Playbook

Operations guide for maintaining and managing the Phishing Email Detector in production.

## Daily Operations

### Health Monitoring

```bash
# Check application health
curl http://your-domain.com/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "database_connected": true,
#   "timestamp": "2024-01-15T10:30:00"
# }
```

### Log Monitoring

```bash
# Docker logs
docker logs -f phishing-detector --tail 100

# Check for errors
docker logs phishing-detector 2>&1 | grep -i error

# Application metrics
curl http://localhost:8000/prometheus-metrics
```

### Performance Metrics

```bash
# Get prediction statistics
curl http://localhost:8000/stats

# Check model metrics
curl http://localhost:8000/metrics
```

## Model Management

### Retraining the Model

When to retrain:
- Model accuracy drops below 88%
- New phishing patterns emerge
- Significant false positive rate
- Scheduled monthly retraining

#### Retraining Process

```bash
# 1. Backup current model
cp models/model-latest.joblib models/model-backup-$(date +%Y%m%d).joblib

# 2. Prepare new training data
# Place updated CSV files in data/raw/

# 3. Train new model
python app/models/trainer.py

# 4. Validate new model
python -c "
import joblib
model = joblib.load('models/model-latest.joblib')
print('Model loaded successfully')
"

# 5. Restart application
docker-compose restart app
# or
systemctl restart phishing-detector
```

### Model Rollback

If new model performs poorly:

```bash
# 1. Stop application
docker-compose stop app

# 2. Restore backup model
cp models/model-backup-YYYYMMDD.joblib models/model-latest.joblib
cp models/feature_extractor-backup-YYYYMMDD.joblib models/feature_extractor-latest.joblib

# 3. Restart application
docker-compose start app

# 4. Verify
curl http://localhost:8000/health
```

### A/B Testing Models

```bash
# Deploy two versions for comparison
# Version A (current)
docker run -d -p 8000:8000 --name phishing-v1 phishing-detector:v1

# Version B (new)
docker run -d -p 8001:8000 --name phishing-v2 phishing-detector:v2

# Use load balancer to split traffic 50/50
# Monitor metrics for both versions
```

## Database Management

### Backup Database

```bash
# Manual backup
DATE=$(date +%Y%m%d_%H%M%S)
cp phishing_detector.db backups/phishing_detector_$DATE.db

# Automated daily backup (add to crontab)
0 2 * * * /path/to/backup_script.sh
```

### Clean Old Data

```bash
# Delete predictions older than 30 days
python << EOF
from app.db import *
from datetime import datetime, timedelta
import asyncio

async def cleanup():
    from sqlalchemy import delete
    from app.db import PredictionHistory, async_engine, AsyncSessionLocal
    
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    
    async with AsyncSessionLocal() as session:
        stmt = delete(PredictionHistory).where(
            PredictionHistory.timestamp < cutoff_date
        )
        await session.execute(stmt)
        await session.commit()
        print(f"Deleted predictions older than {cutoff_date}")

asyncio.run(cleanup())
EOF
```

### Database Migration

```bash
# Backup before migration
cp phishing_detector.db phishing_detector_pre_migration.db

# Run migration scripts
python scripts/migrate_db.py

# Verify
python app/db.py
```

## Scaling Operations

### Scale Up (Horizontal)

```bash
# Docker Compose
docker-compose up --scale app=5

# Kubernetes
kubectl scale deployment phishing-detector --replicas=10

# Verify scaling
kubectl get pods | grep phishing-detector
```

### Scale Down

```bash
# Docker Compose
docker-compose up --scale app=2

# Kubernetes
kubectl scale deployment phishing-detector --replicas=2
```

## Incident Response

### High Error Rate

**Symptoms**: Error rate > 5%

**Actions**:
1. Check logs for error patterns
   ```bash
   docker logs phishing-detector | grep ERROR
   ```

2. Verify model is loaded
   ```bash
   curl http://localhost:8000/health
   ```

3. Check disk space
   ```bash
   df -h
   ```

4. Restart application if needed
   ```bash
   docker-compose restart app
   ```

### High Latency

**Symptoms**: Response time > 2 seconds

**Actions**:
1. Check CPU/Memory usage
   ```bash
   docker stats phishing-detector
   ```

2. Increase workers
   ```bash
   # Update docker-compose.yml
   command: uvicorn app.main:app --workers 4 --host 0.0.0.0
   ```

3. Scale horizontally
   ```bash
   docker-compose up --scale app=3
   ```

### Model Not Found

**Symptoms**: 500 errors on /predict endpoint

**Actions**:
1. Verify model files exist
   ```bash
   ls -la models/model-latest.joblib
   ```

2. Retrain if missing
   ```bash
   python app/models/trainer.py --sample
   ```

3. Restart application
   ```bash
   docker-compose restart app
   ```

### Database Corruption

**Symptoms**: Database connection errors

**Actions**:
1. Stop application
   ```bash
   docker-compose stop app
   ```

2. Restore from backup
   ```bash
   cp backups/phishing_detector_latest.db phishing_detector.db
   ```

3. Verify database integrity
   ```bash
   sqlite3 phishing_detector.db "PRAGMA integrity_check;"
   ```

4. Restart application
   ```bash
   docker-compose start app
   ```

## Maintenance Tasks

### Weekly Tasks

- [ ] Review error logs
- [ ] Check disk usage
- [ ] Verify backups are running
- [ ] Review performance metrics
- [ ] Check for security updates

### Monthly Tasks

- [ ] Review model performance
- [ ] Consider retraining with new data
- [ ] Clean up old prediction data
- [ ] Update dependencies
- [ ] Review and rotate logs
- [ ] Capacity planning review

### Quarterly Tasks

- [ ] Comprehensive security audit
- [ ] Performance benchmarking
- [ ] Disaster recovery drill
- [ ] Update documentation
- [ ] Review SLA compliance

## Monitoring and Alerts

### Set Up Alerts

#### Prometheus Alerts

```yaml
# alerts.yml
groups:
  - name: phishing_detector
    rules:
      - alert: HighErrorRate
        expr: rate(phishing_detector_errors_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, phishing_detector_request_latency_seconds) > 2
        for: 5m
        annotations:
          summary: "High latency detected"
          
      - alert: ModelNotLoaded
        expr: up{job="phishing-detector"} == 0
        for: 1m
        annotations:
          summary: "Application is down"
```

### Email Notifications

```bash
# Configure email alerts
# Add to monitoring system (Grafana, Prometheus Alertmanager, etc.)
```

## Deployment Procedures

### Standard Deployment

```bash
# 1. Pull latest changes
git pull origin main

# 2. Build new image
docker-compose build

# 3. Run tests
pytest

# 4. Deploy with zero downtime
docker-compose up -d --no-deps --build app

# 5. Verify deployment
curl http://localhost:8000/health
```

### Rollback Procedure

```bash
# 1. Stop current version
docker-compose stop app

# 2. Deploy previous version
docker-compose up -d app:previous

# 3. Verify
curl http://localhost:8000/health

# 4. Document rollback reason
echo "Rollback performed on $(date): [REASON]" >> logs/rollback.log
```

## Security Operations

### Update Dependencies

```bash
# Check for vulnerabilities
pip-audit

# Update packages
pip install --upgrade -r requirements.txt

# Test after update
pytest

# Rebuild container
docker-compose build
```

### Rotate Secrets

```bash
# 1. Generate new secrets
openssl rand -base64 32

# 2. Update environment variables
# Edit .env or cloud secret manager

# 3. Restart application
docker-compose restart app
```

### Review Access Logs

```bash
# Check for suspicious activity
docker logs phishing-detector | grep -E "(POST|GET) /predict" | tail -100

# Identify high-frequency IPs
docker logs phishing-detector | grep "POST /predict" | awk '{print $1}' | sort | uniq -c | sort -nr
```

## Troubleshooting Guide

### Problem: Cannot connect to application

**Diagnosis**:
```bash
# Check if container is running
docker ps | grep phishing-detector

# Check container logs
docker logs phishing-detector

# Check port binding
netstat -tulpn | grep 8000
```

**Solution**:
```bash
# Restart container
docker-compose restart app

# Or rebuild if needed
docker-compose up --build -d
```

### Problem: Prediction accuracy degraded

**Diagnosis**:
```bash
# Check model metrics
curl http://localhost:8000/metrics

# Review recent predictions
curl http://localhost:8000/history?limit=100
```

**Solution**:
```bash
# Retrain model with new data
python app/models/trainer.py

# Restart application
docker-compose restart app
```

### Problem: High memory usage

**Diagnosis**:
```bash
# Check memory usage
docker stats phishing-detector --no-stream
```

**Solution**:
```bash
# Set memory limits
docker update --memory 1g --memory-swap 1g phishing-detector

# Or update docker-compose.yml:
# mem_limit: 1g
```

## Performance Tuning

### Optimize Database Queries

```python
# Add indexes for common queries
from app.db import create_engine
engine = create_engine(DATABASE_URL)

# Create indexes
engine.execute("CREATE INDEX idx_timestamp ON prediction_history(timestamp DESC)")
engine.execute("CREATE INDEX idx_label ON prediction_history(label)")
```

### Enable Caching

```python
# Add to app/api/predict.py
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_feature_extraction(text: str):
    return feature_extractor.transform([text])
```

## Contact and Escalation

### On-Call Rotation

- Primary: [Contact Info]
- Secondary: [Contact Info]
- Escalation: [Contact Info]

### Escalation Matrix

| Issue | Severity | Response Time | Escalate To |
|-------|----------|---------------|-------------|
| Service Down | P1 | 15 minutes | Manager |
| High Error Rate | P2 | 1 hour | Tech Lead |
| Performance Degradation | P3 | 4 hours | Team |
| Non-critical Bug | P4 | Next business day | Team |

### Support Channels

- Slack: #phishing-detector
- Email: phishing-detector-ops@company.com
- On-call: [PagerDuty/phone]

## Useful Commands Reference

```bash
# Application
docker-compose up -d              # Start
docker-compose stop               # Stop
docker-compose restart app        # Restart
docker-compose logs -f app        # View logs
docker-compose ps                 # List services

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/stats

# Model operations
python app/models/trainer.py --sample    # Train
ls -lht models/                          # List models

# Database
python app/db.py                         # Initialize
sqlite3 phishing_detector.db             # Access DB

# Testing
pytest                                   # Run all tests
pytest tests/test_api.py -v             # Specific test

# Docker
docker logs phishing-detector            # View logs
docker exec -it phishing-detector bash   # Shell access
docker stats phishing-detector           # Resource usage
```

## Additional Resources

- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [README.md](README_COMPLETE.md) - Project documentation
- [SECURITY.md](SECURITY.md) - Security best practices
