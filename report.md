# ML Tree System - Project Report

## 1. Problem Statement

### Objective
Build an end-to-end machine learning system for binary classification using tree-based models (Decision Tree and Random Forest).

### Dataset
**Breast Cancer Wisconsin (Diagnostic)** - a well-known benchmark dataset for medical diagnosis.

- **Samples**: 569 patients
- **Features**: 30 numeric features (computed from digitized images of breast mass)
- **Target**: Binary classification (malignant: 0, benign: 1)
- **Class distribution**: ~63% benign, ~37% malignant

### Why this dataset?
1. Real-world medical application
2. Clean data with no missing values
3. Good size for demonstration (not too small, not too large)
4. Binary classification suitable for both models
5. Well-documented in scikit-learn

## 2. Methodology

### 2.1 Data Pipeline

```
Load Data → Quality Check → Train/Test Split (80/20) → Preprocessing → Training
```

**Preprocessing steps:**
1. Missing value imputation (median strategy)
2. Standard scaling (zero mean, unit variance)
3. Optional outlier clipping (IQR method)

### 2.2 Models

#### Decision Tree Classifier
- **Algorithm**: CART (Classification and Regression Trees)
- **Splitting criteria**: Gini impurity and Entropy tested
- **Regularization**: max_depth, min_samples_split, min_samples_leaf

#### Random Forest Classifier
- **Algorithm**: Bagging ensemble of decision trees
- **Key parameters**: n_estimators, max_depth, max_features
- **Advantage**: Reduces overfitting through averaging

### 2.3 Hyperparameter Tuning

Used **RandomizedSearchCV** with:
- 5-fold stratified cross-validation
- 50 random combinations per model
- F1-score as optimization metric (handles class imbalance)

**Search spaces:**

| Parameter | Decision Tree | Random Forest |
|-----------|--------------|---------------|
| max_depth | [3, 5, 7, 10, None] | [5, 10, 15, None] |
| min_samples_split | [2, 5, 10] | [2, 5, 10] |
| min_samples_leaf | [1, 2, 4] | [1, 2, 4] |
| n_estimators | N/A | [50, 100, 200] |
| max_features | N/A | [sqrt, log2] |
| criterion | [gini, entropy] | N/A |

## 3. Results

### 3.1 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **0.965** | **0.964** | **0.965** | **0.964** | **0.992** |
| Decision Tree | 0.939 | 0.940 | 0.939 | 0.939 | 0.936 |

**Key observations:**
- Random Forest outperforms Decision Tree on all metrics
- Both models achieve >93% accuracy
- High ROC-AUC indicates good discrimination ability
- Random Forest shows ~5.6% improvement in ROC-AUC

### 3.2 Best Hyperparameters

**Decision Tree:**
```
max_depth: 7
min_samples_split: 5
min_samples_leaf: 2
criterion: entropy
```

**Random Forest:**
```
n_estimators: 100
max_depth: 10
min_samples_split: 2
min_samples_leaf: 1
max_features: sqrt
```

### 3.3 Feature Importance (Random Forest)

Top 10 most important features:

1. worst perimeter (0.142)
2. worst concave points (0.138)
3. worst radius (0.121)
4. mean concave points (0.089)
5. worst area (0.079)
6. mean perimeter (0.052)
7. mean concavity (0.050)
8. mean radius (0.048)
9. mean area (0.045)
10. worst concavity (0.041)

**Insight**: "Worst" (largest) tumor measurements and concave points are most predictive of malignancy.

### 3.4 Error Analysis

**Decision Tree errors:**
- Total: ~7 misclassifications on test set
- Higher false negatives (missing malignant cases)
- Errors clustered on borderline cases

**Random Forest errors:**
- Total: ~4 misclassifications on test set
- More balanced error distribution
- Better handling of edge cases

## 4. Explainability

### 4.1 Decision Tree Visualization
- Full tree exported as PNG and GraphViz DOT file
- Shows decision rules at each node
- Interpretable: "IF worst_perimeter > 105.5 THEN likely malignant"

### 4.2 Feature Importance
- Random Forest provides robust importance scores
- Consistent with medical literature (tumor size and shape matter)
- Top 5 features account for ~57% of predictive power

### 4.3 Confusion Matrices
- Visual representation of classification errors
- Random Forest shows fewer false negatives (critical in medical context)

## 5. Reproducibility

### Measures taken:
1. **Random seeds**: Fixed at 42 for all random operations
2. **Model persistence**: Saved with joblib (includes fitted parameters)
3. **Pipeline persistence**: Preprocessing pipeline saved for consistent transforms
4. **Metadata storage**: Feature names, target names, best parameters archived
5. **Configuration file**: All settings in YAML for easy tracking

### To reproduce results:
```bash
python main.py train --config config/config.yaml
```

## 6. Deployment

### REST API (FastAPI)
- **Endpoint**: POST /predict
- **Input**: 30 feature values as JSON array
- **Output**: Prediction, probability, class name
- **Validation**: Pydantic models for input validation
- **Documentation**: Auto-generated Swagger UI at /docs

### Example request:
```json
POST /predict
{
  "features": [17.99, 10.38, ...],
  "model_name": "random_forest"
}
```

### Example response:
```json
{
  "prediction": 0,
  "class_name": "malignant",
  "probability": 0.95,
  "model_used": "random_forest"
}
```

## 7. Limitations

1. **Dataset size**: Only 569 samples - larger dataset would improve generalization
2. **Binary classification**: System designed for 2 classes only
3. **Feature engineering**: Minimal feature engineering performed
4. **No deep learning**: Tree-based models may miss complex non-linear patterns
5. **Class imbalance**: ~63/37 split - could benefit from SMOTE or weighted classes
6. **No time-series**: Assumes independent samples (no temporal patterns)

## 8. Next Steps

### Short-term improvements:
1. Add more preprocessing options (feature selection, PCA)
2. Implement SHAP values for better explainability
3. Add Gradient Boosting (XGBoost, LightGBM) for comparison
4. Implement A/B testing for model selection

### Long-term enhancements:
1. Model monitoring and drift detection
2. Automated retraining pipeline
3. Docker containerization
4. Kubernetes deployment
5. Integration with MLflow for experiment tracking

## 9. Conclusion

This project demonstrates a complete ML workflow from data loading to API deployment. Random Forest achieved 96.5% accuracy with strong interpretability through feature importance. The modular design and comprehensive testing make it suitable for production use after addressing the noted limitations.

---

**Author**: ML Tree System
**Version**: 1.0.0
**Date**: 2025
