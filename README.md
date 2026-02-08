# ML Tree System

End-to-end Machine Learning system using Decision Tree and Random Forest classifiers.

## Overview

This project implements a production-ready ML pipeline for binary classification using the Breast Cancer Wisconsin dataset. It demonstrates best practices in:

- Data pipeline design
- Model training with hyperparameter tuning
- Model evaluation and comparison
- Explainability and error analysis
- REST API deployment
- Software engineering practices

## Project Structure

```
ml_tree_system/
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   └── raw/                  # Raw data directory
├── models/                   # Saved models and artifacts
├── reports/
│   └── figures/              # Generated visualizations
├── src/
│   ├── data/                 # Data loading module
│   ├── features/             # Preprocessing pipeline
│   ├── models/               # Training and evaluation
│   ├── visualization/        # Plotting functions
│   └── utils/                # Helper functions
├── api/
│   └── app.py                # FastAPI application
├── tests/                    # Unit tests
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── report.md                 # Project report
```

## Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train both Decision Tree and Random Forest models:

```bash
python main.py train --config config/config.yaml
```

This will:
- Load the breast cancer dataset
- Preprocess data with standardization
- Perform hyperparameter tuning with cross-validation
- Evaluate and compare both models
- Generate visualizations
- Save models and artifacts to `models/`

### Prediction

Make predictions using a trained model:

```bash
# From CSV file
python main.py predict --model models/random_forest.joblib --input data/sample.csv --output predictions.csv

# Example output to console
python main.py predict --model models/random_forest.joblib --input data/sample.csv
```

### REST API

Start the prediction API server:

```bash
python main.py api --port 8000
```

Or directly with uvicorn:
```bash
uvicorn api.app:app --reload --port 8000
```

API endpoints:
- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /features` - List expected features
- `GET /models` - List available models
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
                 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
                 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
                 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
    "model_name": "random_forest"
  }'
```

Example response:
```json
{
  "prediction": 0,
  "class_name": "malignant",
  "probability": 0.95,
  "probabilities": {"malignant": 0.95, "benign": 0.05},
  "model_used": "random_forest"
}
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## Configuration

Edit `config/config.yaml` to customize:

- Random seed for reproducibility
- Train/test split ratio
- Preprocessing options (scaler type, imputer strategy)
- Hyperparameter search space
- Cross-validation settings
- API host and port

## Models

### Decision Tree
- Interpretable white-box model
- Visualizable decision rules
- Prone to overfitting without pruning

### Random Forest
- Ensemble of decision trees
- More robust and accurate
- Provides feature importance
- Less interpretable than single tree

## Outputs

After training, the following artifacts are generated:

### Models (`models/`)
- `decision_tree.joblib` - Trained Decision Tree
- `random_forest.joblib` - Trained Random Forest
- `preprocessor.joblib` - Fitted preprocessing pipeline
- `metadata.joblib` - Feature names and best parameters
- `model_comparison.csv` - Performance comparison table
- `feature_importance.csv` - Feature importance rankings

### Figures (`reports/figures/`)
- `confusion_matrix_dt.png` - Decision Tree confusion matrix
- `confusion_matrix_rf.png` - Random Forest confusion matrix
- `feature_importance_rf.png` - Top feature importance chart
- `decision_tree.png` - Tree structure visualization
- `roc_comparison.png` - ROC curve comparison
- `model_comparison.png` - Performance bar chart

## Performance

Typical results on Breast Cancer dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.965 | 0.964 | 0.965 | 0.964 | 0.992 |
| Decision Tree | 0.939 | 0.940 | 0.939 | 0.939 | 0.936 |

## License

MIT License
