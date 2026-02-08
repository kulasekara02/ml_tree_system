# Local Development & Testing Guide

## Option 1: Docker Compose (Recommended for Local)

### Prerequisites
- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)

### Quick Start

```bash
# Build and start the container
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Access the API
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Option 2: Docker Run

```bash
# Build image
docker build -t ml-tree-system:local .

# Run container
docker run -p 8000:8000 -v ${PWD}/models:/app/models ml-tree-system:local

# Run with all volumes
docker run -p 8000:8000 \
  -v ${PWD}/config:/app/config \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/logs:/app/logs \
  ml-tree-system:local
```

## Option 3: Local Python

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Train models first
python main.py train

# Run API
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Testing the API

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/info

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "random_forest",
    "features": {
      "mean_radius": 17.99,
      "mean_texture": 10.38,
      "mean_perimeter": 122.8,
      "mean_area": 1001.0,
      "mean_smoothness": 0.1184,
      "mean_compactness": 0.2776,
      "mean_concavity": 0.3001,
      "mean_concave_points": 0.1471,
      "mean_symmetry": 0.2419,
      "mean_fractal_dimension": 0.07871,
      "radius_error": 1.095,
      "texture_error": 0.9053,
      "perimeter_error": 8.589,
      "area_error": 153.4,
      "smoothness_error": 0.006399,
      "compactness_error": 0.04904,
      "concavity_error": 0.05373,
      "concave_points_error": 0.01587,
      "symmetry_error": 0.03003,
      "fractal_dimension_error": 0.006193,
      "worst_radius": 25.38,
      "worst_texture": 17.33,
      "worst_perimeter": 184.6,
      "worst_area": 2019.0,
      "worst_smoothness": 0.1622,
      "worst_compactness": 0.6656,
      "worst_concavity": 0.7119,
      "worst_concave_points": 0.2654,
      "worst_symmetry": 0.4601,
      "worst_fractal_dimension": 0.1189
    }
  }'
```

### Using Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
data = {
    "model": "random_forest",
    "features": {
        "mean_radius": 17.99,
        # ... other features
    }
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Kubernetes Deployment

### Prerequisites for K8s
- Kubernetes cluster (minikube, kind, or cloud provider)
- kubectl configured

### Option A: Minikube (Local K8s Cluster)

```bash
# Install minikube (Windows)
choco install minikube

# Start minikube
minikube start

# Build image in minikube
minikube image build -t ml-tree-system:local .

# Apply K8s configs
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml

# Update deployment to use local image
# Edit k8s/deployment.yaml: change image to ml-tree-system:local
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Access service
minikube service ml-tree-system -n ml-system

# View dashboard
minikube dashboard
```

### Option B: Kind (Kubernetes in Docker)

```bash
# Install kind
choco install kind

# Create cluster
kind create cluster --name ml-cluster

# Load image
kind load docker-image ml-tree-system:local --name ml-cluster

# Apply configs
kubectl apply -f k8s/
```

### Option C: Cloud Kubernetes (GKE, EKS, AKS)

See [DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment instructions.

## CI/CD Setup

### GitHub Actions will automatically:
1. Run tests on every push/PR
2. Build Docker image on main branch
3. Push to Docker Hub
4. Deploy to K8s cluster (if configured)

### Required GitHub Secrets:
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub token
- `KUBE_CONFIG`: Base64 kubeconfig (for auto-deploy)

## Troubleshooting

### Docker build fails
```bash
# Clear Docker cache
docker system prune -a

# Build without cache
docker build --no-cache -t ml-tree-system:local .
```

### Port already in use
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /PID <PID> /F
```

### Models not found
```bash
# Train models first
python main.py train

# Or copy pre-trained models to models/ directory
```

### API not responding
```bash
# Check container logs
docker-compose logs -f

# Check if container is running
docker ps

# Test health endpoint
curl http://localhost:8000/health
```

## Performance Testing

```bash
# Install Apache Bench
choco install apache-httpd

# Simple load test
ab -n 1000 -c 10 http://localhost:8000/health

# Or use hey
choco install hey
hey -n 1000 -c 10 http://localhost:8000/health
```

## Monitoring

### View Container Stats
```bash
docker stats ml-tree-system

# Or with docker-compose
docker-compose stats
```

### View Application Logs
```bash
# Docker Compose
docker-compose logs -f --tail=100

# Docker
docker logs -f ml-tree-system

# Kubernetes
kubectl logs -f deployment/ml-tree-system -n ml-system
```
