# Deployment Guide

This document provides detailed instructions for deploying the Phishing Email Detector to various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [Google Cloud Platform](#google-cloud-platform-gcp)
  - [Amazon Web Services](#amazon-web-services-aws)
  - [Microsoft Azure](#microsoft-azure)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker and Docker Compose installed
- Cloud CLI tools (gcloud, aws-cli, or az-cli)
- Model artifacts (train using `python app/models/trainer.py --sample`)
- SSL certificate for production

## Local Deployment

### Using Python

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python app/models/trainer.py --sample

# 4. Run application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Using Docker Compose

```bash
# Build and start
docker-compose up --build

# Access at http://localhost:8000
```

## Docker Deployment

### Build Image

```bash
docker build -t phishing-detector:latest -f docker/Dockerfile .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e LOG_LEVEL=INFO \
  --name phishing-detector \
  phishing-detector:latest
```

### Persist Data with Volumes

```bash
# Create volumes
docker volume create phishing-models
docker volume create phishing-data

# Run with volumes
docker run -d \
  -p 8000:8000 \
  -v phishing-models:/app/models \
  -v phishing-data:/app/data \
  --restart unless-stopped \
  --name phishing-detector \
  phishing-detector:latest
```

## Cloud Deployment

### Google Cloud Platform (GCP)

#### Option 1: Cloud Run (Recommended)

```bash
# 1. Set project
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/phishing-detector

# 3. Deploy to Cloud Run
gcloud run deploy phishing-detector \
  --image gcr.io/YOUR_PROJECT_ID/phishing-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 2 \
  --max-instances 10
```

#### Option 2: Google Kubernetes Engine (GKE)

```bash
# 1. Create cluster
gcloud container clusters create phishing-detector-cluster \
  --num-nodes 2 \
  --machine-type n1-standard-2 \
  --region us-central1

# 2. Deploy application
kubectl apply -f infra/kubernetes/deployment.yaml
kubectl apply -f infra/kubernetes/service.yaml
```

### Amazon Web Services (AWS)

#### Option 1: ECS Fargate

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name phishing-detector

# 2. Build and push image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag phishing-detector:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/phishing-detector:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/phishing-detector:latest

# 3. Deploy with Terraform
cd infra/terraform/aws
terraform init
terraform apply
```

#### Option 2: EC2 Instance

```bash
# SSH into EC2 instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# Run container
docker run -d -p 80:8000 \
  --restart always \
  your-registry/phishing-detector:latest
```

### Microsoft Azure

#### Option 1: Container Instances

```bash
# 1. Create resource group
az group create --name phishing-detector-rg --location eastus

# 2. Create container registry
az acr create --resource-group phishing-detector-rg \
  --name phishingdetectoracr --sku Basic

# 3. Push image
az acr login --name phishingdetectoracr
docker tag phishing-detector:latest phishingdetectoracr.azurecr.io/phishing-detector:latest
docker push phishingdetectoracr.azurecr.io/phishing-detector:latest

# 4. Deploy container instance
az container create \
  --resource-group phishing-detector-rg \
  --name phishing-detector \
  --image phishingdetectoracr.azurecr.io/phishing-detector:latest \
  --cpu 2 --memory 2 \
  --registry-login-server phishingdetectoracr.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label phishing-detector \
  --ports 8000
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```env
# Application
LOG_LEVEL=INFO
JSON_LOGS=true

# Model
MODEL_PATH=models/model-latest.joblib
FEATURE_EXTRACTOR_PATH=models/feature_extractor-latest.joblib

# Database
DATABASE_URL=sqlite+aiosqlite:///phishing_detector.db

# Security (Production)
ALLOWED_ORIGINS=https://yourdomain.com
API_KEY_REQUIRED=true
RATE_LIMIT_PER_MINUTE=60
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

### SSL/TLS Certificate

```bash
# Using Let's Encrypt
sudo certbot --nginx -d yourdomain.com
```

## Monitoring

### Health Checks

```bash
# Check application health
curl http://localhost:8000/health

# Check Prometheus metrics
curl http://localhost:8000/prometheus-metrics
```

### Docker Health Check

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' phishing-detector
```

### Cloud Monitoring

#### GCP Cloud Monitoring

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=phishing-detector" --limit 50
```

#### AWS CloudWatch

```bash
# View logs
aws logs tail /aws/ecs/phishing-detector --follow
```

## Troubleshooting

### Common Issues

#### Model Not Found

```bash
# Solution: Train the model
python app/models/trainer.py --sample

# Verify model files exist
ls -la models/
```

#### Database Connection Error

```bash
# Solution: Initialize database
python app/db.py

# Check permissions
chmod 664 phishing_detector.db
```

#### Container Won't Start

```bash
# Check logs
docker logs phishing-detector

# Run interactively for debugging
docker run -it --rm phishing-detector:latest /bin/bash
```

#### High Memory Usage

```bash
# Solution: Limit container memory
docker run -d -p 8000:8000 \
  --memory="1g" \
  --memory-swap="1g" \
  phishing-detector:latest
```

### Performance Optimization

#### Increase Workers

```bash
# For production, use multiple workers
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Enable Caching

Add model caching to reduce load times:

```python
# In app/api/predict.py
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    # Model loading code
    pass
```

## Scaling

### Horizontal Scaling

```bash
# Docker Compose - Scale to 3 instances
docker-compose up --scale app=3

# Kubernetes - Scale deployment
kubectl scale deployment phishing-detector --replicas=5
```

### Auto-scaling

#### GCP Cloud Run

Cloud Run auto-scales by default based on requests.

#### AWS ECS

```bash
# Configure auto-scaling with Terraform
# See infra/terraform/aws/autoscaling.tf
```

## Backup and Recovery

### Backup Database

```bash
# Backup SQLite database
cp phishing_detector.db phishing_detector_backup_$(date +%Y%m%d).db

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp phishing_detector.db backups/phishing_detector_$DATE.db
find backups/ -mtime +7 -delete  # Keep only 7 days of backups
```

### Model Versioning

Models are automatically versioned with timestamps. Keep multiple versions:

```bash
# List all model versions
ls -lht models/model-v*.joblib

# Rollback to previous version
cp models/model-v20240115_103000.joblib models/model-latest.joblib
```

## Cost Optimization

### Cloud Run (GCP)

- Use minimum instances: 0 (scale to zero)
- Set max instances based on expected load
- Estimated cost: $5-50/month for 10K-100K requests

### ECS Fargate (AWS)

- Use Fargate Spot for non-critical workloads
- Right-size CPU and memory allocations
- Estimated cost: $20-100/month for 2 tasks

### Container Instances (Azure)

- Stop instances during low-traffic hours
- Use consumption-based pricing
- Estimated cost: $10-60/month

## Security Checklist

- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Implement rate limiting
- [ ] Use secret management (not env files)
- [ ] Enable container scanning
- [ ] Set up Web Application Firewall (WAF)
- [ ] Implement authentication for sensitive endpoints
- [ ] Regular security updates
- [ ] Monitor for suspicious activity
- [ ] Backup data regularly

## Next Steps

1. Set up monitoring and alerting
2. Configure auto-scaling policies
3. Implement CI/CD pipeline
4. Set up staging environment
5. Create disaster recovery plan
6. Document runbooks for operations team

For operations procedures, see [OPERATIONS.md](OPERATIONS.md).
