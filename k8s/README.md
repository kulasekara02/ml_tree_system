# Kubernetes Deployment Guide

## Prerequisites
- Kubernetes cluster (v1.20+)
- kubectl configured
- Docker Hub account
- Helm (optional, for easier management)

## Quick Start

### 1. Create Namespace
```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Create ConfigMap and PVC
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
```

### 3. Deploy Application
```bash
# Update the image in deployment.yaml with your Docker Hub username
kubectl apply -f k8s/deployment.yaml
```

### 4. Expose Service
```bash
kubectl apply -f k8s/service.yaml
```

### 5. Configure Ingress (Optional)
```bash
# Update the host in ingress.yaml
kubectl apply -f k8s/ingress.yaml
```

### 6. Enable Auto-scaling
```bash
kubectl apply -f k8s/hpa.yaml
```

## Verify Deployment

```bash
# Check pods
kubectl get pods -n ml-system

# Check services
kubectl get svc -n ml-system

# Check deployment
kubectl get deployment -n ml-system

# View logs
kubectl logs -f deployment/ml-tree-system -n ml-system
```

## Access the Application

```bash
# Get external IP
kubectl get svc ml-tree-system -n ml-system

# Port forward for local testing
kubectl port-forward svc/ml-tree-system 8000:80 -n ml-system
```

## Update Deployment

```bash
# Update image
kubectl set image deployment/ml-tree-system ml-tree-system=your-username/ml-tree-system:new-tag -n ml-system

# Check rollout status
kubectl rollout status deployment/ml-tree-system -n ml-system

# Rollback if needed
kubectl rollout undo deployment/ml-tree-system -n ml-system
```

## Monitoring

```bash
# Watch pods
kubectl get pods -n ml-system -w

# Check HPA status
kubectl get hpa -n ml-system

# View events
kubectl get events -n ml-system --sort-by='.lastTimestamp'
```

## Cleanup

```bash
kubectl delete namespace ml-system
```
