# GitHub Actions CI/CD Setup Guide

## ✅ Automatic Triggers

Your GitHub Actions pipeline runs automatically on:
- Every push to `main` branch
- Every push to `develop` branch  
- Every pull request to `main` branch

## 🔐 Required Setup (One-time)

### Step 1: Get Docker Hub Token

1. Go to https://hub.docker.com/settings/security
2. Click "New Access Token"
3. Name it: `github-actions`
4. Copy the token (save it - you won't see it again!)

### Step 2: Add GitHub Secrets

1. Go to your repository: https://github.com/kulasekara02/ml_tree_system
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add:

| Secret Name | Value |
|------------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | The token from Step 1 |
| `KUBE_CONFIG` | (Optional) Base64 encoded kubeconfig for K8s deployment |

### Step 3: Trigger the Pipeline

The pipeline runs automatically on every push! To trigger it now:

```bash
# Make a small change and push
git commit --allow-empty -m "Trigger CI/CD pipeline"
git push
```

## 📊 View Pipeline Status

1. Go to: https://github.com/kulasekara02/ml_tree_system/actions
2. Click on the latest workflow run
3. Watch the progress:
   - ✅ **Test** - Runs pytest with coverage
   - ✅ **Build** - Builds Docker image
   - ✅ **Deploy** - Deploys to Kubernetes (if secrets configured)

## 🎯 What Happens in Each Stage

### 1️⃣ Test Stage (Runs on all branches)
```yaml
- Installs Python dependencies
- Runs pytest tests
- Generates coverage report
- Uploads to codecov (optional)
```

### 2️⃣ Build Stage (After tests pass)
```yaml
- Builds Docker image
- Pushes to Docker Hub as:
  - username/ml-tree-system:latest
  - username/ml-tree-system:<commit-sha>
```

### 3️⃣ Deploy Stage (Only on main branch)
```yaml
- Connects to Kubernetes cluster
- Updates deployment with new image
- Verifies rollout success
```

## 🚨 Troubleshooting

### Pipeline Fails on Build Stage
**Issue**: Docker Hub secrets not configured
**Solution**: Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets

### Pipeline Fails on Deploy Stage  
**Issue**: Kubernetes not configured
**Solution**: 
- Either add `KUBE_CONFIG` secret
- Or remove deploy job from `.github/workflows/ci-cd.yml`

### Tests Fail
**Issue**: Code has errors
**Solution**: 
```bash
# Run tests locally first
pytest tests/ -v

# Fix any failing tests
# Commit and push
git add .
git commit -m "Fix tests"
git push
```

## 🎉 Success Indicators

✅ Green checkmark on commits in GitHub
✅ Docker image available at: `https://hub.docker.com/r/YOUR_USERNAME/ml-tree-system`
✅ Can pull and run: `docker pull YOUR_USERNAME/ml-tree-system:latest`

## 📝 Skip CI/CD (if needed)

Add `[skip ci]` to your commit message:
```bash
git commit -m "Update README [skip ci]"
```

## 🔄 Manual Workflow Trigger

You can also manually trigger workflows:
1. Go to: https://github.com/kulasekara02/ml_tree_system/actions
2. Select "CI/CD Pipeline"
3. Click "Run workflow"
4. Choose branch and click "Run workflow"
