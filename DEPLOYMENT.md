# Deployment Guide - ML Tree System

## CI/CD Pipeline Setup

### GitHub Actions Secrets Required

Add these secrets to your GitHub repository (Settings → Secrets → Actions):

1. **DOCKER_USERNAME**: Your Docker Hub username
2. **DOCKER_PASSWORD**: Your Docker Hub access token
3. **KUBE_CONFIG**: Base64 encoded Kubernetes config file

```bash
# Encode your kubeconfig
cat ~/.kube/config | base64 -w 0
```

### Pipeline Stages

The CI/CD pipeline consists of three stages:

#### 1. **Test**
- Runs on every push and pull request
- Installs dependencies
- Executes pytest with coverage
- Uploads coverage reports

#### 2. **Build**
- Runs after tests pass
- Builds Docker image
- Pushes to Docker Hub with tags:
  - `latest`
  - Commit SHA (for rollback)

#### 3. **Deploy**
- Runs only on main branch
- Updates Kubernetes deployment
- Rolls out new version
- Verifies deployment

## Manual Deployment

### 1. Build Docker Image Locally

```bash
docker build -t your-username/ml-tree-system:latest .
docker push your-username/ml-tree-system:latest
```

### 2. Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml

# Update image in deployment.yaml first
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Optional: Ingress
kubectl apply -f k8s/ingress.yaml
```

### 3. Verify Deployment

```bash
# Check all resources
kubectl get all -n ml-system

# Check logs
kubectl logs -f deployment/ml-tree-system -n ml-system

# Test health endpoint
kubectl port-forward svc/ml-tree-system 8000:80 -n ml-system
curl http://localhost:8000/health
```

## Local Development with Docker

```bash
# Build
docker build -t ml-tree-system:dev .

# Run
docker run -p 8000:8000 ml-tree-system:dev

# Run with volume mount for development
docker run -p 8000:8000 -v $(pwd):/app ml-tree-system:dev
```

## Monitoring & Debugging

### View Logs
```bash
kubectl logs -f deployment/ml-tree-system -n ml-system
```

### Execute Commands in Pod
```bash
kubectl exec -it deployment/ml-tree-system -n ml-system -- /bin/bash
```

### Check Resource Usage
```bash
kubectl top pods -n ml-system
kubectl top nodes
```

### Describe Resources
```bash
kubectl describe deployment ml-tree-system -n ml-system
kubectl describe pod <pod-name> -n ml-system
```

## Scaling

### Manual Scaling
```bash
kubectl scale deployment ml-tree-system --replicas=5 -n ml-system
```

### Auto-scaling (HPA is configured)
- Min replicas: 2
- Max replicas: 10
- Target CPU: 70%
- Target Memory: 80%

## Rollback

```bash
# View revision history
kubectl rollout history deployment/ml-tree-system -n ml-system

# Rollback to previous version
kubectl rollout undo deployment/ml-tree-system -n ml-system

# Rollback to specific revision
kubectl rollout undo deployment/ml-tree-system --to-revision=2 -n ml-system
```

## Security Best Practices

1. Use specific image tags instead of `latest` in production
2. Run containers as non-root user
3. Use secrets for sensitive data
4. Enable network policies
5. Regular security scans of images

## Cost Optimization

1. Set appropriate resource requests/limits
2. Use HPA for dynamic scaling
3. Enable cluster autoscaler
4. Use spot instances for non-critical workloads

## Troubleshooting

### Pod not starting
```bash
kubectl describe pod <pod-name> -n ml-system
kubectl logs <pod-name> -n ml-system
```

### Service not accessible
```bash
kubectl get endpoints -n ml-system
kubectl get svc -n ml-system
```

### Image pull errors
```bash
# Check if image exists
docker pull your-username/ml-tree-system:latest

# Create image pull secret if needed
kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=<your-username> \
  --docker-password=<your-password> \
  -n ml-system
```
