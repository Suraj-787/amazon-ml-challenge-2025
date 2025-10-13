# Docker Setup for ML Challenge 2025 - Multimodal Fusion

**Team:** Monk Builders  
**Members:** Suraj Kumar, Himanshu Rai, Pratik Gautam, Impana KS

This repository contains a Dockerized ML solution using **uv** as the package manager for fast and reliable dependency management.

## 🐳 Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- At least 4GB RAM available for Docker
- GPU support (optional, for CUDA acceleration)

## 📁 Project Structure

```
submission_files/
├── Dockerfile              # Main Docker configuration
├── docker-compose.yml      # Multi-service orchestration
├── .dockerignore           # Files to exclude from build
├── pyproject.toml          # Python project configuration
├── uv.lock                 # Locked dependencies
├── multimodal_fusion.py    # Main ML pipeline
├── sample_code.py          # Sample predictor
├── train.csv              # Training data
├── test.csv               # Test data
└── output/                # Generated results
```

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
# Build the image
docker-compose build ml-challenge

# Or build manually
docker build -t multimodal-fusion .
```

### 2. Run the Multimodal Fusion Pipeline

```bash
# Run the complete ML pipeline
docker-compose up ml-challenge

# Or run manually
docker run --rm -v $(pwd):/app multimodal-fusion
```

### 3. Development Mode

```bash
# Start interactive development container
docker-compose run --rm ml-dev

# Inside the container, you can:
uv run python multimodal_fusion.py
uv run python sample_code.py
uv add new-package  # Add new dependencies
```

## 🛠️ Available Services

### Main ML Pipeline
```bash
docker-compose up ml-challenge
```
Runs the complete multimodal fusion pipeline with XGBoost and LightGBM.

### Development Shell
```bash
docker-compose run --rm ml-dev
```
Interactive container for development and debugging.

### Sample Predictor
```bash
docker-compose up sample-predictor
```
Runs the basic sample prediction code.

### Jupyter Notebook (Optional)
```bash
docker-compose up jupyter
```
Starts Jupyter Lab at `http://localhost:8888` for interactive development.

## 📊 Data Management

### Input Data
Place your data files in the project root:
- `train.csv` - Training dataset
- `test.csv` - Test dataset

### Output Data
Results are saved to:
- `multimodal_predictions.csv` - Final predictions
- `multimodal_xgboost_model.pkl` - Trained XGBoost model
- `multimodal_lightgbm_model.pkl` - Trained LightGBM model

### External Data Directory
```bash
# Mount external data directory
docker-compose run --rm -v /path/to/your/data:/app/data ml-challenge
```

## ⚙️ Configuration

### Environment Variables

```bash
# Set in docker-compose.yml or pass to docker run
PYTHONUNBUFFERED=1     # Unbuffered Python output
PYTHONPATH=/app        # Python path
MAX_FEATURES=10000     # Max TF-IDF features
TEST_SIZE=0.2          # Validation split size
```

### Memory and CPU Limits

```yaml
# In docker-compose.yml
services:
  ml-challenge:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
```

## 🚄 GPU Support (Optional)

For CUDA acceleration, update `docker-compose.yml`:

```yaml
services:
  ml-challenge:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Then run:
```bash
docker-compose up ml-challenge
```

## 📦 Package Management with UV

### Add New Dependencies
```bash
# Enter development container
docker-compose run --rm ml-dev

# Add packages
uv add pandas numpy scikit-learn
uv add --dev pytest jupyter

# Lock dependencies
uv lock

# Sync environment
uv sync
```

### Update Dependencies
```bash
uv lock --upgrade
docker-compose build ml-challenge  # Rebuild image
```

## 🔍 Monitoring and Logs

### View Logs
```bash
# Follow logs in real-time
docker-compose logs -f ml-challenge

# View specific service logs
docker-compose logs sample-predictor
```

### Resource Usage
```bash
# Monitor container resources
docker stats multimodal-fusion

# View container processes
docker-compose top
```

## 🧪 Testing

### Run Tests
```bash
# Run basic validation
docker-compose run --rm ml-dev uv run python -m pytest

# Test sample predictor
docker-compose run --rm sample-predictor
```

### Validate Data
```bash
# Check data format
docker-compose run --rm ml-dev uv run python -c "
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print(f'Train columns: {train.columns.tolist()}')
print(f'Test columns: {test.columns.tolist()}')
"
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Increase Docker memory limit
   docker run --memory=8g --rm multimodal-fusion
   ```

2. **Permission Errors**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

3. **UV Not Found**
   ```bash
   # Rebuild image to install uv
   docker-compose build --no-cache ml-challenge
   ```

4. **CUDA Issues**
   ```bash
   # Check CUDA availability
   docker run --rm --gpus all multimodal-fusion uv run python -c "import torch; print(torch.cuda.is_available())"
   ```

### Debug Mode
```bash
# Run with debug output
docker-compose run --rm -e PYTHONPATH=/app -e DEBUG=1 ml-dev
```

## 📈 Performance Optimization

### Build Optimization
```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build

# Multi-stage builds (if needed)
docker build --target production -t multimodal-fusion .
```

### Runtime Optimization
```bash
# Use specific CPU/memory limits
docker run --cpus="4" --memory="8g" multimodal-fusion
```

## 🚢 Deployment

### Production Build
```bash
# Build for production
docker build -t multimodal-fusion:latest .

# Push to registry
docker tag multimodal-fusion:latest your-registry/multimodal-fusion:latest
docker push your-registry/multimodal-fusion:latest
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multimodal-fusion
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: ml-pipeline
        image: multimodal-fusion:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

## 📚 Additional Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

## 🤝 Support

For issues and questions:
1. Check the logs: `docker-compose logs ml-challenge`
2. Validate your data format
3. Ensure sufficient system resources
4. Review environment variables

Happy coding! 🚀

---
**Developed by Team Monk Builders**  
Suraj Kumar, Himanshu Rai, Pratik Gautam, Impana KS